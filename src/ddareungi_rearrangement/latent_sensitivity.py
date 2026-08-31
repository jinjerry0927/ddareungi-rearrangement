from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib
import polars as pl

from ddareungi_rearrangement.latent_demand import (
    CONTRACT_VERSION,
    SCENARIO_ORDER,
    LatentDemandConfig,
    LatentDemandError,
    build_latent_demand_manifest,
)
from ddareungi_rearrangement.simulation import (
    GreedyNearestPolicy,
    NoRelocationPolicy,
    ReplayScenario,
    SimulationConfig,
    SimulationError,
    SimulationRun,
    build_replay_scenario,
    simulate_replay,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FROZEN_SEEDS = tuple(range(20251124, 20251174))
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_OBSERVED_CONTRACT = {
    "requests": 16_205,
    "p0_failed_rentals": 1_905,
    "p2r_failed_rentals": 691,
}


class LatentSensitivityError(RuntimeError):
    """잠재수요 paired batch 입력이나 불변조건이 유효하지 않을 때 발생하는 오류."""


@dataclass(frozen=True)
class LatentSensitivityResults:
    observed_baseline: tuple[dict[str, Any], ...]
    policy_runs: pl.DataFrame
    paired_runs: pl.DataFrame
    scenario_summary: pl.DataFrame
    station_runs: pl.DataFrame
    station_summary: pl.DataFrame
    station_robustness: dict[str, Any]
    stop_checks: dict[str, dict[str, Any]]
    stop_required: bool


@dataclass(frozen=True)
class LatentSensitivityExperiment:
    generated_at_utc: str
    method: str
    contract_version: str
    training_window: str
    evaluation_window: str
    stations: int
    seeds: tuple[int, ...]
    scenarios: tuple[str, ...]
    policies: dict[str, dict[str, Any]]
    observed_baseline: tuple[dict[str, Any], ...]
    scenario_summary: tuple[dict[str, Any], ...]
    station_robustness: dict[str, Any]
    stop_checks: dict[str, dict[str, Any]]
    stop_required: bool
    interpretation_status: str
    limitations: tuple[str, ...]
    output_files: dict[str, str]


def _protected_policy(
    coordinates: dict[str, tuple[float, float]],
) -> GreedyNearestPolicy:
    return GreedyNearestPolicy(
        coordinates=coordinates,
        donor_reserve_bikes=7,
        max_actions_per_decision=3,
        vehicle_capacity=10,
        average_speed_kmh=20.0,
        road_distance_factor=1.3,
        handling_minutes_per_bike=0.75,
        name="p2r_protected_reserve_7",
    )


def _run_paired_policies(
    scenario: ReplayScenario,
    coordinates: dict[str, tuple[float, float]],
) -> tuple[tuple[str, SimulationRun], ...]:
    return (
        ("P0", simulate_replay(scenario, NoRelocationPolicy())),
        ("P2-R", simulate_replay(scenario, _protected_policy(coordinates))),
    )


def _station_p10(source_station_metrics: pl.DataFrame, request_source: str) -> float:
    source_rows = source_station_metrics.filter(pl.col("request_source") == request_source)
    eligible = source_rows.filter(pl.col("requests") >= 10).sort("service_rate")
    if eligible.is_empty():
        eligible = source_rows.filter(pl.col("requests") > 0).sort("service_rate")
    if eligible.is_empty():
        return 0.0
    rates = eligible["service_rate"].sort().to_list()
    return float(rates[min(len(rates) - 1, int(len(rates) * 0.1))])


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    if not values:
        raise LatentSensitivityError("bootstrap 입력이 비어 있습니다")
    point = mean(values)
    if len(values) == 1:
        return point, point, point
    rng = random.Random(seed)
    sample_size = len(values)
    estimates = sorted(
        mean(values[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    )
    lower = estimates[int(resamples * 0.025)]
    upper = estimates[min(resamples - 1, int(resamples * 0.975))]
    return point, lower, upper


def _validate_run(
    run: SimulationRun,
    *,
    expected_hash: str,
    expected_synthetic_requests: int,
    label: str,
) -> None:
    source = run.source_metrics
    stations = run.source_station_metrics
    if source is None or stations is None:
        raise LatentSensitivityError(f"{label} 출처별 지표가 생성되지 않았습니다")
    if source.request_manifest_hash != expected_hash:
        raise LatentSensitivityError(f"{label} manifest hash가 paired 입력과 다릅니다")
    if source.synthetic_requests != expected_synthetic_requests:
        raise LatentSensitivityError(f"{label} 합성 요청 수가 manifest와 다릅니다")
    if source.observed_requests + source.synthetic_requests != source.combined_requests:
        raise LatentSensitivityError(f"{label} 관측·합성 요청 합이 combined와 다릅니다")
    if (
        source.observed_successful_rentals + source.observed_failed_rentals
        != source.observed_requests
        or source.synthetic_successful_rentals + source.synthetic_failed_rentals
        != source.synthetic_requests
        or source.combined_successful_rentals + source.combined_failed_rentals
        != source.combined_requests
    ):
        raise LatentSensitivityError(f"{label} 출처별 성공·실패 합이 요청 수와 다릅니다")
    if any(
        residual != 0
        for residual in (
            source.observed_trip_flow_residual,
            source.synthetic_trip_flow_residual,
            source.combined_trip_flow_residual,
            source.total_conservation_residual,
            run.metrics.conservation_residual,
        )
    ):
        raise LatentSensitivityError(f"{label} 보존식 잔차가 0이 아닙니다")


def _baseline_rows(
    scenario: ReplayScenario,
    coordinates: dict[str, tuple[float, float]],
    expected_contract: dict[str, int] | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[SimulationRun, SimulationRun]]:
    paired = _run_paired_policies(scenario, coordinates)
    p0 = paired[0][1]
    p2r = paired[1][1]
    if p0.metrics.observed_requests != p2r.metrics.observed_requests:
        raise LatentSensitivityError("관측 전용 P0와 P2-R 요청 수가 다릅니다")
    if p0.metrics.conservation_residual != 0 or p2r.metrics.conservation_residual != 0:
        raise LatentSensitivityError("관측 전용 실행의 자전거 보존식 잔차가 0이 아닙니다")
    if expected_contract is not None:
        actual = {
            "requests": p0.metrics.observed_requests,
            "p0_failed_rentals": p0.metrics.failed_rentals,
            "p2r_failed_rentals": p2r.metrics.failed_rentals,
        }
        if actual != expected_contract:
            raise LatentSensitivityError(
                "관측 전용 홀드아웃 계약값이 바뀌었습니다: "
                f"expected={expected_contract}, actual={actual}"
            )
    station_effects = p0.station_metrics.select(
        "station_id", pl.col("failed_rentals").alias("p0_failed")
    ).join(
        p2r.station_metrics.select("station_id", pl.col("failed_rentals").alias("p2r_failed")),
        on="station_id",
        how="inner",
        validate="1:1",
    )
    worsened = int((station_effects["p2r_failed"] > station_effects["p0_failed"]).sum())
    rows = tuple(
        {
            "policy_role": role,
            "requests": run.metrics.observed_requests,
            "successful_rentals": run.metrics.successful_rentals,
            "failed_rentals": run.metrics.failed_rentals,
            "service_rate": run.metrics.service_rate,
            "p10_station_service_rate": run.metrics.p10_station_service_rate,
            "conservation_residual": run.metrics.conservation_residual,
            "stations_worsened_vs_p0": 0 if role == "P0" else worsened,
        }
        for role, run in paired
    )
    return rows, (p0, p2r)


def _summarize_paired_runs(paired_runs: pl.DataFrame) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        rows = paired_runs.filter(pl.col("scenario") == scenario).sort("seed")
        if rows.is_empty():
            continue
        combined_values = [float(value) for value in rows["combined_failures_avoided"].to_list()]
        observed_values = [float(value) for value in rows["observed_failures_avoided"].to_list()]
        service_values = [
            float(value) for value in rows["combined_service_rate_pp_change"].to_list()
        ]
        p10_values = [float(value) for value in rows["combined_p10_pp_change"].to_list()]
        combined_point, combined_low, combined_high = _bootstrap_mean_ci(
            combined_values,
            seed=20251124 + scenario_index * 10,
        )
        observed_point, observed_low, observed_high = _bootstrap_mean_ci(
            observed_values,
            seed=20251125 + scenario_index * 10,
        )
        service_point, service_low, service_high = _bootstrap_mean_ci(
            service_values,
            seed=20251126 + scenario_index * 10,
        )
        p10_point, p10_low, p10_high = _bootstrap_mean_ci(
            p10_values,
            seed=20251127 + scenario_index * 10,
        )
        records.append(
            {
                "scenario": scenario,
                "seeds": rows.height,
                "synthetic_requests_mean": round(mean(rows["synthetic_requests"]), 3),
                "synthetic_requests_min": int(rows["synthetic_requests"].min()),
                "synthetic_requests_max": int(rows["synthetic_requests"].max()),
                "synthetic_share_mean": round(mean(rows["synthetic_share"]), 6),
                "combined_failures_avoided_mean": round(combined_point, 3),
                "combined_failures_avoided_median": round(median(combined_values), 3),
                "combined_failures_avoided_ci95_low": round(combined_low, 3),
                "combined_failures_avoided_ci95_high": round(combined_high, 3),
                "observed_failures_avoided_mean": round(observed_point, 3),
                "observed_failures_avoided_ci95_low": round(observed_low, 3),
                "observed_failures_avoided_ci95_high": round(observed_high, 3),
                "combined_service_rate_pp_change_mean": round(service_point, 4),
                "combined_service_rate_pp_change_ci95_low": round(service_low, 4),
                "combined_service_rate_pp_change_ci95_high": round(service_high, 4),
                "combined_p10_pp_change_mean": round(p10_point, 4),
                "combined_p10_pp_change_ci95_low": round(p10_low, 4),
                "combined_p10_pp_change_ci95_high": round(p10_high, 4),
                "stations_worsened_mean": round(mean(rows["stations_worsened_vs_p0"]), 3),
                "stations_worsened_min": int(rows["stations_worsened_vs_p0"].min()),
                "stations_worsened_max": int(rows["stations_worsened_vs_p0"].max()),
            }
        )
    return pl.DataFrame(records)


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise LatentSensitivityError("대여소 분위수 입력이 비어 있습니다")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return int(ordered[index])


def _station_sort_key(station_id: str) -> tuple[bool, int | str]:
    return (not station_id.isdigit(), int(station_id) if station_id.isdigit() else station_id)


def _summarize_station_runs(
    station_runs: pl.DataFrame,
    coordinates: dict[str, tuple[float, float]],
    *,
    expected_seed_count: int,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        for request_source in ("observed", "synthetic_latent", "combined"):
            for station_id in sorted(coordinates, key=_station_sort_key):
                rows = station_runs.filter(
                    (pl.col("scenario") == scenario)
                    & (pl.col("request_source") == request_source)
                    & (pl.col("station_id") == station_id)
                ).sort("seed")
                if rows.height != expected_seed_count:
                    raise LatentSensitivityError(
                        f"{scenario}/{request_source}/{station_id} seed 행 수가 다릅니다"
                    )
                failures_avoided = [int(value) for value in rows["failures_avoided"].to_list()]
                worsening_seeds = sum(value < 0 for value in failures_avoided)
                improving_seeds = sum(value > 0 for value in failures_avoided)
                worsening_probability = worsening_seeds / expected_seed_count
                if worsening_probability >= 0.8:
                    stability_class = "persistent_worsened"
                elif worsening_probability > 0:
                    stability_class = "intermittent_worsened"
                else:
                    stability_class = "never_worsened"
                baseline_failures_avoided = int(rows["observed_baseline_failures_avoided"].item(0))
                latitude, longitude = coordinates[station_id]
                records.append(
                    {
                        "scenario": scenario,
                        "request_source": request_source,
                        "station_id": station_id,
                        "station_name": str(rows["station_name"].item(0)),
                        "latitude": latitude,
                        "longitude": longitude,
                        "seeds": expected_seed_count,
                        "requests_mean": round(mean(rows["requests"].to_list()), 3),
                        "failures_avoided_mean": round(mean(failures_avoided), 3),
                        "failures_avoided_median": round(median(failures_avoided), 3),
                        "failures_avoided_p10": _nearest_rank(failures_avoided, 0.1),
                        "failures_avoided_p90": _nearest_rank(failures_avoided, 0.9),
                        "worsening_seeds": worsening_seeds,
                        "worsening_probability": round(worsening_probability, 6),
                        "improving_seeds": improving_seeds,
                        "improving_probability": round(
                            improving_seeds / expected_seed_count,
                            6,
                        ),
                        "tied_seeds": expected_seed_count - worsening_seeds - improving_seeds,
                        "p0_service_rate_mean": round(mean(rows["p0_service_rate"].to_list()), 6),
                        "p2r_service_rate_mean": round(mean(rows["p2r_service_rate"].to_list()), 6),
                        "observed_baseline_failures_avoided": baseline_failures_avoided,
                        "observed_baseline_worsened": baseline_failures_avoided < 0,
                        "new_worsening_under_latent": (
                            baseline_failures_avoided >= 0 and worsening_probability > 0
                        ),
                        "stability_class": stability_class,
                    }
                )
    summary = pl.DataFrame(records)
    if set(summary["station_id"].unique().to_list()) != set(coordinates):
        raise LatentSensitivityError("대여소 공간 요약과 좌표 범위가 다릅니다")
    return summary


def _build_station_robustness(station_summary: pl.DataFrame) -> dict[str, Any]:
    combined = station_summary.filter(pl.col("request_source") == "combined")
    scenario_findings: dict[str, dict[str, Any]] = {}
    persistent_sets: list[set[str]] = []
    for scenario in SCENARIO_ORDER:
        rows = combined.filter(pl.col("scenario") == scenario)
        persistent = rows.filter(pl.col("stability_class") == "persistent_worsened")
        intermittent = rows.filter(pl.col("stability_class") == "intermittent_worsened")
        never = rows.filter(pl.col("stability_class") == "never_worsened")
        persistent_ids = set(persistent["station_id"].to_list())
        persistent_sets.append(persistent_ids)
        top_worsened = (
            rows.filter(pl.col("worsening_probability") > 0)
            .sort(
                ["worsening_probability", "failures_avoided_mean", "station_id"],
                descending=[True, False, False],
            )
            .select(
                "station_id",
                "station_name",
                "worsening_probability",
                "failures_avoided_mean",
                "failures_avoided_p10",
                "failures_avoided_p90",
                "observed_baseline_worsened",
                "new_worsening_under_latent",
            )
            .head(10)
            .to_dicts()
        )
        scenario_findings[scenario] = {
            "persistent_worsened_stations": persistent.height,
            "intermittent_worsened_stations": intermittent.height,
            "never_worsened_stations": never.height,
            "observed_baseline_worsened_stations": rows.filter(
                pl.col("observed_baseline_worsened")
            ).height,
            "persistent_observed_baseline_worsened": persistent.filter(
                pl.col("observed_baseline_worsened")
            ).height,
            "new_worsening_stations": rows.filter(pl.col("new_worsening_under_latent")).height,
            "persistent_new_worsening_stations": persistent.filter(
                pl.col("new_worsening_under_latent")
            ).height,
            "top_worsened": top_worsened,
        }
    persistent_all = set.intersection(*persistent_sets) if persistent_sets else set()
    return {
        "classification": {
            "persistent_worsened": "worsening_probability >= 0.8",
            "intermittent_worsened": "0 < worsening_probability < 0.8",
            "never_worsened": "worsening_probability == 0",
        },
        "scenarios": scenario_findings,
        "persistent_all_scenarios_count": len(persistent_all),
        "persistent_all_scenarios_station_ids": sorted(
            persistent_all,
            key=_station_sort_key,
        ),
    }


def _build_stop_checks(
    paired_runs: pl.DataFrame,
    observed_baseline: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    baseline = {str(row["policy_role"]): row for row in observed_baseline}
    baseline_failure_direction = _sign(
        float(baseline["P0"]["failed_rentals"] - baseline["P2-R"]["failed_rentals"])
    )
    baseline_p10_direction = _sign(
        float(
            baseline["P2-R"]["p10_station_service_rate"]
            - baseline["P0"]["p10_station_service_rate"]
        )
    )
    baseline_has_worsened = bool(baseline["P2-R"]["stations_worsened_vs_p0"] > 0)
    combined_directions = {
        _sign(float(value)) for value in paired_runs["combined_failures_avoided"].to_list()
    }
    p10_directions = {
        _sign(float(value)) for value in paired_runs["combined_p10_pp_change"].to_list()
    }
    worsened_states = {
        bool(value > 0) for value in paired_runs["stations_worsened_vs_p0"].to_list()
    }
    observed_combined_mismatches = paired_runs.filter(
        pl.col("observed_failure_direction") != pl.col("combined_failure_direction")
    ).height
    max_backoff = float(paired_runs["district_fallback_rate"].max())
    max_hard_cap = float(paired_runs["hard_capped_interval_rate"].max())
    max_synthetic_share = float(paired_runs["synthetic_share"].max())
    return {
        "policy_failure_direction_change": {
            "triggered": combined_directions != {baseline_failure_direction},
            "observed_baseline_direction": baseline_failure_direction,
            "latent_directions": sorted(combined_directions),
        },
        "observed_combined_direction_mismatch": {
            "triggered": observed_combined_mismatches > 0,
            "mismatched_seed_scenarios": observed_combined_mismatches,
        },
        "p10_direction_change": {
            "triggered": p10_directions != {baseline_p10_direction},
            "observed_baseline_direction": baseline_p10_direction,
            "latent_directions": sorted(p10_directions),
        },
        "worsened_station_conclusion_change": {
            "triggered": worsened_states != {baseline_has_worsened},
            "observed_baseline_has_worsened": baseline_has_worsened,
            "latent_states": sorted(worsened_states),
        },
        "district_backoff_over_5pct": {
            "triggered": max_backoff > 0.05,
            "maximum_rate": max_backoff,
            "threshold": 0.05,
        },
        "hard_cap_at_least_1pct": {
            "triggered": max_hard_cap >= 0.01,
            "maximum_rate": max_hard_cap,
            "threshold": 0.01,
        },
        "synthetic_share_over_25pct": {
            "triggered": max_synthetic_share > 0.25,
            "maximum_rate": max_synthetic_share,
            "threshold": 0.25,
        },
    }


def run_latent_demand_sensitivity(
    trips: pl.DataFrame,
    station_hour: pl.DataFrame,
    coordinates: dict[str, tuple[float, float]],
    *,
    training_start: datetime,
    training_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    expected_observed_contract: dict[str, int] | None = None,
    enforce_frozen_seed_set: bool = False,
) -> LatentSensitivityResults:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed는 중복 없는 하나 이상의 값이어야 합니다")
    if enforce_frozen_seed_set and seeds != FROZEN_SEEDS:
        raise ValueError("최종 batch는 동결된 seed 20251124..20251173만 사용할 수 있습니다")
    config = SimulationConfig(
        start=evaluation_start,
        end=evaluation_end,
        decision_interval_minutes=60,
        max_bikes_per_decision=40,
    )
    observed_scenario = build_replay_scenario(
        trips,
        station_hour,
        config,
        eligible_station_ids=set(coordinates),
    )
    observed_baseline, baseline_runs = _baseline_rows(
        observed_scenario,
        coordinates,
        expected_observed_contract,
    )
    baseline_station_effects = (
        baseline_runs[0]
        .station_metrics.select("station_id", pl.col("failed_rentals").alias("p0_failed"))
        .join(
            baseline_runs[1].station_metrics.select(
                "station_id", pl.col("failed_rentals").alias("p2r_failed")
            ),
            on="station_id",
            how="inner",
            validate="1:1",
        )
        .with_columns((pl.col("p0_failed") - pl.col("p2r_failed")).alias("failures_avoided"))
    )
    baseline_failures_by_station = {
        str(row["station_id"]): int(row["failures_avoided"])
        for row in baseline_station_effects.iter_rows(named=True)
    }
    observed_request_count = baseline_runs[0].metrics.observed_requests
    policy_records: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    station_records: list[dict[str, Any]] = []
    for seed in seeds:
        manifest = build_latent_demand_manifest(
            trips,
            station_hour,
            LatentDemandConfig(
                training_start=training_start,
                training_end=training_end,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                seed=seed,
            ),
            eligible_station_ids=set(coordinates),
        )
        scenario_ids = {
            scenario: set(manifest.requests_for(scenario)["request_id"].to_list())
            for scenario in SCENARIO_ORDER
        }
        if not scenario_ids["low"] <= scenario_ids["base"] <= scenario_ids["high"]:
            raise LatentSensitivityError(f"seed {seed}의 low/base/high 요청이 중첩되지 않습니다")
        for scenario in SCENARIO_ORDER:
            latent_requests = manifest.requests_for(scenario)
            manifest_hash = manifest.audit.scenario_manifest_hashes[scenario]
            replay = build_replay_scenario(
                trips,
                station_hour,
                config,
                eligible_station_ids=set(coordinates),
                latent_requests=latent_requests,
                latent_request_manifest_hash=manifest_hash,
            )
            paired = _run_paired_policies(replay, coordinates)
            runs = {role: run for role, run in paired}
            for role, run in paired:
                _validate_run(
                    run,
                    expected_hash=manifest_hash,
                    expected_synthetic_requests=latent_requests.height,
                    label=f"seed={seed}, scenario={scenario}, policy={role}",
                )
                source = run.source_metrics
                source_stations = run.source_station_metrics
                if source is None or source_stations is None:
                    raise LatentSensitivityError("검증 뒤 출처별 지표가 사라졌습니다")
                policy_records.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "policy_role": role,
                        "manifest_hash": manifest_hash,
                        "district_fallback_rate": manifest.audit.district_fallback_rate,
                        "hard_capped_interval_rate": manifest.audit.hard_capped_interval_rate,
                        "intensity_rate_capped_interval_rate": (
                            manifest.audit.intensity_rate_capped_interval_rate
                        ),
                        "dropped_candidates_by_hard_cap": (
                            manifest.audit.dropped_candidates_by_hard_cap
                        ),
                        **asdict(source),
                        "observed_p10_service_rate": _station_p10(source_stations, "observed"),
                        "synthetic_p10_service_rate": _station_p10(
                            source_stations, "synthetic_latent"
                        ),
                        "combined_p10_service_rate": _station_p10(source_stations, "combined"),
                        "empty_station_hours": run.metrics.empty_station_hours,
                        "bikes_moved": run.metrics.bikes_moved,
                        "relocation_distance_km": run.metrics.relocation_distance_km,
                        "relocation_vehicle_minutes": run.metrics.relocation_vehicle_minutes,
                    }
                )
            p0 = runs["P0"]
            p2r = runs["P2-R"]
            p0_source = p0.source_metrics
            p2r_source = p2r.source_metrics
            p0_stations = p0.source_station_metrics
            p2r_stations = p2r.source_station_metrics
            if any(item is None for item in (p0_source, p2r_source, p0_stations, p2r_stations)):
                raise LatentSensitivityError("paired 출처별 지표가 누락됐습니다")
            assert p0_source is not None and p2r_source is not None
            assert p0_stations is not None and p2r_stations is not None
            if (
                p0_source.request_manifest_hash != p2r_source.request_manifest_hash
                or p0_source.combined_requests != p2r_source.combined_requests
                or p0_source.observed_requests != p2r_source.observed_requests
                or p0_source.synthetic_requests != p2r_source.synthetic_requests
            ):
                raise LatentSensitivityError(
                    f"seed={seed}, scenario={scenario}의 paired hash·요청 수가 다릅니다"
                )
            expected_failures_avoided = {
                "observed": (
                    p0_source.observed_failed_rentals - p2r_source.observed_failed_rentals
                ),
                "synthetic_latent": (
                    p0_source.synthetic_failed_rentals - p2r_source.synthetic_failed_rentals
                ),
                "combined": (
                    p0_source.combined_failed_rentals - p2r_source.combined_failed_rentals
                ),
            }
            for request_source in ("observed", "synthetic_latent", "combined"):
                source_effects = (
                    p0_stations.filter(pl.col("request_source") == request_source)
                    .select(
                        "station_id",
                        "station_name",
                        pl.col("requests").alias("p0_requests"),
                        pl.col("failed_rentals").alias("p0_failed_rentals"),
                        pl.col("service_rate").alias("p0_service_rate"),
                    )
                    .join(
                        p2r_stations.filter(pl.col("request_source") == request_source).select(
                            "station_id",
                            pl.col("requests").alias("p2r_requests"),
                            pl.col("failed_rentals").alias("p2r_failed_rentals"),
                            pl.col("service_rate").alias("p2r_service_rate"),
                        ),
                        on="station_id",
                        how="inner",
                        validate="1:1",
                    )
                    .with_columns(
                        (pl.col("p0_failed_rentals") - pl.col("p2r_failed_rentals")).alias(
                            "failures_avoided"
                        )
                    )
                )
                if source_effects.height != len(coordinates):
                    raise LatentSensitivityError(
                        f"seed={seed}, scenario={scenario}, source={request_source} "
                        "대여소 수가 다릅니다"
                    )
                if source_effects.filter(pl.col("p0_requests") != pl.col("p2r_requests")).height:
                    raise LatentSensitivityError(
                        f"seed={seed}, scenario={scenario}, source={request_source} "
                        "요청 수가 다릅니다"
                    )
                station_reconciliation = int(source_effects["failures_avoided"].sum())
                if station_reconciliation != expected_failures_avoided[request_source]:
                    raise LatentSensitivityError(
                        f"seed={seed}, scenario={scenario}, source={request_source} "
                        "대여소 실패 차이 합이 전체와 다릅니다"
                    )
                for row in source_effects.iter_rows(named=True):
                    station_id = str(row["station_id"])
                    station_records.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "request_source": request_source,
                            "manifest_hash": manifest_hash,
                            "station_id": station_id,
                            "station_name": str(row["station_name"]),
                            "requests": int(row["p0_requests"]),
                            "p0_failed_rentals": int(row["p0_failed_rentals"]),
                            "p2r_failed_rentals": int(row["p2r_failed_rentals"]),
                            "failures_avoided": int(row["failures_avoided"]),
                            "p0_service_rate": float(row["p0_service_rate"]),
                            "p2r_service_rate": float(row["p2r_service_rate"]),
                            "observed_baseline_failures_avoided": (
                                baseline_failures_by_station[station_id]
                            ),
                        }
                    )
            station_effects = p0.station_metrics.select(
                "station_id", pl.col("failed_rentals").alias("p0_failed")
            ).join(
                p2r.station_metrics.select(
                    "station_id", pl.col("failed_rentals").alias("p2r_failed")
                ),
                on="station_id",
                how="inner",
                validate="1:1",
            )
            failures_avoided = station_effects["p0_failed"] - station_effects["p2r_failed"]
            observed_failures_avoided = (
                p0_source.observed_failed_rentals - p2r_source.observed_failed_rentals
            )
            combined_failures_avoided = (
                p0_source.combined_failed_rentals - p2r_source.combined_failed_rentals
            )
            p0_combined_p10 = _station_p10(p0_stations, "combined")
            p2r_combined_p10 = _station_p10(p2r_stations, "combined")
            paired_records.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "manifest_hash": manifest_hash,
                    "observed_requests": p0_source.observed_requests,
                    "synthetic_requests": p0_source.synthetic_requests,
                    "combined_requests": p0_source.combined_requests,
                    "synthetic_share": (
                        p0_source.synthetic_requests / observed_request_count
                        if observed_request_count
                        else 0.0
                    ),
                    "p0_observed_failed_rentals": p0_source.observed_failed_rentals,
                    "p2r_observed_failed_rentals": p2r_source.observed_failed_rentals,
                    "observed_failures_avoided": observed_failures_avoided,
                    "p0_combined_failed_rentals": p0_source.combined_failed_rentals,
                    "p2r_combined_failed_rentals": p2r_source.combined_failed_rentals,
                    "combined_failures_avoided": combined_failures_avoided,
                    "observed_failure_direction": _sign(observed_failures_avoided),
                    "combined_failure_direction": _sign(combined_failures_avoided),
                    "combined_service_rate_pp_change": round(
                        (p2r_source.combined_service_rate - p0_source.combined_service_rate) * 100,
                        6,
                    ),
                    "combined_p10_pp_change": round(
                        (p2r_combined_p10 - p0_combined_p10) * 100,
                        6,
                    ),
                    "stations_improved_vs_p0": int((failures_avoided > 0).sum()),
                    "stations_tied_vs_p0": int((failures_avoided == 0).sum()),
                    "stations_worsened_vs_p0": int((failures_avoided < 0).sum()),
                    "district_fallback_rate": manifest.audit.district_fallback_rate,
                    "hard_capped_interval_rate": manifest.audit.hard_capped_interval_rate,
                }
            )
    policy_frame = pl.DataFrame(policy_records).sort("seed", "scenario", "policy_role")
    paired_frame = pl.DataFrame(paired_records).sort("seed", "scenario")
    scenario_summary = _summarize_paired_runs(paired_frame)
    station_frame = pl.DataFrame(station_records).sort(
        "seed", "scenario", "request_source", "station_id"
    )
    station_summary = _summarize_station_runs(
        station_frame,
        coordinates,
        expected_seed_count=len(seeds),
    )
    station_robustness = _build_station_robustness(station_summary)
    stop_checks = _build_stop_checks(paired_frame, observed_baseline)
    return LatentSensitivityResults(
        observed_baseline=observed_baseline,
        policy_runs=policy_frame,
        paired_runs=paired_frame,
        scenario_summary=scenario_summary,
        station_runs=station_frame,
        station_summary=station_summary,
        station_robustness=station_robustness,
        stop_checks=stop_checks,
        stop_required=any(bool(check["triggered"]) for check in stop_checks.values()),
    )


def _plot_summary(summary: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = summary["scenario"].to_list()
    positions = list(range(len(scenarios)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    combined = summary["combined_failures_avoided_mean"].to_list()
    combined_low = summary["combined_failures_avoided_ci95_low"].to_list()
    combined_high = summary["combined_failures_avoided_ci95_high"].to_list()
    axes[0].errorbar(
        positions,
        combined,
        yerr=[
            [point - low for point, low in zip(combined, combined_low, strict=True)],
            [high - point for point, high in zip(combined, combined_high, strict=True)],
        ],
        fmt="o",
        capsize=5,
        color="#1479ff",
    )
    axes[0].axhline(0, color="#666666", linewidth=1)
    axes[0].set_title("P2-R combined failures avoided")
    axes[0].set_ylabel("requests (mean, 95% bootstrap CI)")
    p10 = summary["combined_p10_pp_change_mean"].to_list()
    p10_low = summary["combined_p10_pp_change_ci95_low"].to_list()
    p10_high = summary["combined_p10_pp_change_ci95_high"].to_list()
    axes[1].errorbar(
        positions,
        p10,
        yerr=[
            [point - low for point, low in zip(p10, p10_low, strict=True)],
            [high - point for point, high in zip(p10, p10_high, strict=True)],
        ],
        fmt="o",
        capsize=5,
        color="#e05a33",
    )
    axes[1].axhline(0, color="#666666", linewidth=1)
    axes[1].set_title("P2-R combined p10 change")
    axes[1].set_ylabel("percentage points (mean, 95% bootstrap CI)")
    for axis in axes:
        axis.set_xticks(positions, scenarios)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Gangnam latent-demand sensitivity (50 paired seeds)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_station_robustness(station_summary: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = station_summary.filter(pl.col("request_source") == "combined")
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
    scatter = None
    for axis, scenario in zip(axes, SCENARIO_ORDER, strict=True):
        rows = combined.filter(pl.col("scenario") == scenario)
        scatter = axis.scatter(
            rows["longitude"],
            rows["latitude"],
            c=rows["worsening_probability"],
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            s=38,
            alpha=0.9,
            edgecolors="#555555",
            linewidths=0.3,
        )
        baseline_worsened = rows.filter(pl.col("observed_baseline_worsened"))
        axis.scatter(
            baseline_worsened["longitude"],
            baseline_worsened["latitude"],
            facecolors="none",
            edgecolors="#1565c0",
            linewidths=1.2,
            s=80,
            label="observed-baseline worsened",
        )
        top_rows = (
            rows.filter(pl.col("worsening_probability") > 0)
            .sort(
                ["worsening_probability", "failures_avoided_mean"],
                descending=[True, False],
            )
            .head(3)
        )
        for row in top_rows.iter_rows(named=True):
            axis.annotate(
                str(row["station_id"]),
                (float(row["longitude"]), float(row["latitude"])),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(scenario)
        axis.set_xlabel("longitude")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("latitude")
    axes[0].legend(loc="lower left", fontsize=8)
    if scatter is None:
        raise LatentSensitivityError("대여소 공간 강건성 지도 입력이 비어 있습니다")
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.12, top=0.86, wspace=0.08)
    colorbar_axis = figure.add_axes((0.9, 0.2, 0.014, 0.58))
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label("P2-R worsening probability across 50 seeds")
    figure.suptitle("Gangnam station robustness under latent-demand scenarios")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_markdown(experiment: LatentSensitivityExperiment) -> str:
    baseline_rows = "\n".join(
        "| {role} | {requests:,} | {failed:,} | {service:.2%} | {p10:.2%} | {worse} |".format(
            role=row["policy_role"],
            requests=row["requests"],
            failed=row["failed_rentals"],
            service=row["service_rate"],
            p10=row["p10_station_service_rate"],
            worse=row["stations_worsened_vs_p0"],
        )
        for row in experiment.observed_baseline
    )
    summary_rows = "\n".join(
        "| {scenario} | {synthetic:,.1f} ({minimum:,}~{maximum:,}) | "
        "{combined:,.1f} [{combined_low:,.1f}, {combined_high:,.1f}] | "
        "{observed:,.1f} [{observed_low:,.1f}, {observed_high:,.1f}] | "
        "{p10:+.3f} [{p10_low:+.3f}, {p10_high:+.3f}] | {worse:.1f} |".format(
            scenario=row["scenario"],
            synthetic=row["synthetic_requests_mean"],
            minimum=row["synthetic_requests_min"],
            maximum=row["synthetic_requests_max"],
            combined=row["combined_failures_avoided_mean"],
            combined_low=row["combined_failures_avoided_ci95_low"],
            combined_high=row["combined_failures_avoided_ci95_high"],
            observed=row["observed_failures_avoided_mean"],
            observed_low=row["observed_failures_avoided_ci95_low"],
            observed_high=row["observed_failures_avoided_ci95_high"],
            p10=row["combined_p10_pp_change_mean"],
            p10_low=row["combined_p10_pp_change_ci95_low"],
            p10_high=row["combined_p10_pp_change_ci95_high"],
            worse=row["stations_worsened_mean"],
        )
        for row in experiment.scenario_summary
    )
    station_rows = "\n".join(
        "| {scenario} | {persistent} | {intermittent} | {new} | {new_persistent} |".format(
            scenario=scenario,
            persistent=experiment.station_robustness["scenarios"][scenario][
                "persistent_worsened_stations"
            ],
            intermittent=experiment.station_robustness["scenarios"][scenario][
                "intermittent_worsened_stations"
            ],
            new=experiment.station_robustness["scenarios"][scenario]["new_worsening_stations"],
            new_persistent=experiment.station_robustness["scenarios"][scenario][
                "persistent_new_worsening_stations"
            ],
        )
        for scenario in SCENARIO_ORDER
    )
    high_hotspots = "\n".join(
        "| {station_id} {station_name} | {probability:.0%} | {mean:+.2f} | {p10:+}~{p90:+} | "
        "{baseline} |".format(
            station_id=row["station_id"],
            station_name=row["station_name"],
            probability=row["worsening_probability"],
            mean=row["failures_avoided_mean"],
            p10=row["failures_avoided_p10"],
            p90=row["failures_avoided_p90"],
            baseline="기존 악화" if row["observed_baseline_worsened"] else "잠재수요 새 악화",
        )
        for row in experiment.station_robustness["scenarios"]["high"]["top_worsened"][:5]
    )
    persistent_station_ids = (
        ", ".join(experiment.station_robustness["persistent_all_scenarios_station_ids"]) or "없음"
    )
    check_rows = "\n".join(
        f"| `{name}` | {'TRIGGERED' if check['triggered'] else 'PASS'} |"
        for name, check in experiment.stop_checks.items()
    )
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    conclusion = (
        "중단조건이 감지돼 정책 결론을 내리지 않는다. 아래 감사값을 확인해야 한다."
        if experiment.stop_required
        else "사전 중단조건은 감지되지 않았다. 결과는 인과효과가 아닌 사후 민감도 범위다."
    )
    return f"""# 강남구 잠재수요 v2 정책 민감도

## 결론

- 판정: `{experiment.interpretation_status}`
- {conclusion}
- 실험: {len(experiment.seeds)}개 공통 seed × {len(experiment.scenarios)}개 시나리오 × 2개 정책
- 계약: `{experiment.contract_version}`

## 관측 전용 기준선

| 정책 | 요청 | 실패 | 성공률 | p10 | P0보다 악화된 대여소 |
|---|---:|---:|---:|---:|---:|
{baseline_rows}

## 잠재수요 paired 결과

실패 방지는 P0 실패 - P2-R 실패다. 대괄호는 seed 50개를 동일 가중한 평균의 결정론적
95% bootstrap 신뢰구간이다.

| 수준 | 합성요청 평균 (범위) | combined 실패방지 | 관측 실패방지 | p10 변화(%p) | 악화지점 평균 |
|---|---:|---:|---:|---:|---:|
{summary_rows}

## 대여소 공간 강건성

`persistent`는 50개 seed 중 80% 이상에서 P2-R 실패가 P0보다 많은 대여소다. `new`는 관측
전용 기준에서는 악화되지 않았지만 해당 잠재수요 수준에서 한 번 이상 악화된 대여소다.

| 수준 | persistent 악화 | intermittent 악화 | 새 악화 | 새 persistent 악화 |
|---|---:|---:|---:|---:|
{station_rows}

- 세 수준 모두 persistent 악화: {experiment.station_robustness["persistent_all_scenarios_count"]}곳
- 대여소 ID: {persistent_station_ids}

### high 수준 악화확률 상위 5곳

실패 차이는 `P0 실패 - P2-R 실패`이므로 음수일수록 P2-R에서 더 악화됐다.

| 대여소 | 악화확률 | 평균 실패방지 | p10~p90 | 구분 |
|---|---:|---:|---:|---|
{high_hotspots}

## 사전 중단조건

| 검사 | 판정 |
|---|---|
{check_rows}

## 실험 계약

- 학습기간: {experiment.training_window}
- 평가기간: {experiment.evaluation_window}
- 대여소: {experiment.stations}개
- seed: {experiment.seeds[0]}..{experiment.seeds[-1]}
- P2-R: donor reserve 7, 시간당 최대 3회, 평균 20km/h, 적재 10대
- 모든 seed·수준에서 P0와 P2-R은 같은 manifest hash와 초기 재고를 사용

## 해석 제한

{limitations}
"""


def build_latent_demand_sensitivity(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    training_start: datetime,
    training_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
    policy_runs_csv_path: Path,
    paired_runs_csv_path: Path,
    summary_csv_path: Path,
    station_runs_parquet_path: Path,
    station_summary_csv_path: Path,
    figure_path: Path,
    station_figure_path: Path,
    json_path: Path,
    markdown_path: Path,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    expected_observed_contract: dict[str, int] | None = EXPECTED_OBSERVED_CONTRACT,
    enforce_frozen_seed_set: bool = True,
) -> LatentSensitivityExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise LatentSensitivityError(f"잠재수요 민감도 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise LatentSensitivityError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise LatentSensitivityError("좌표 대여소 ID가 중복됐습니다")
    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    try:
        results = run_latent_demand_sensitivity(
            trips,
            station_hour,
            coordinates,
            training_start=training_start,
            training_end=training_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            seeds=seeds,
            expected_observed_contract=expected_observed_contract,
            enforce_frozen_seed_set=enforce_frozen_seed_set,
        )
    except (LatentDemandError, SimulationError) as exc:
        raise LatentSensitivityError(f"잠재수요 시뮬레이션 실패: {exc}") from exc
    for path in (
        policy_runs_csv_path,
        paired_runs_csv_path,
        summary_csv_path,
        station_runs_parquet_path,
        station_summary_csv_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    results.policy_runs.write_csv(policy_runs_csv_path)
    results.paired_runs.write_csv(paired_runs_csv_path)
    results.scenario_summary.write_csv(summary_csv_path)
    results.station_runs.write_parquet(station_runs_parquet_path, compression="zstd")
    results.station_summary.write_csv(station_summary_csv_path)
    _plot_summary(results.scenario_summary, figure_path)
    _plot_station_robustness(results.station_summary, station_figure_path)
    output_files = {
        "policy_runs": str(policy_runs_csv_path),
        "paired_runs": str(paired_runs_csv_path),
        "scenario_summary": str(summary_csv_path),
        "station_runs": str(station_runs_parquet_path),
        "station_summary": str(station_summary_csv_path),
        "figure": str(figure_path),
        "station_figure": str(station_figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = LatentSensitivityExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="post_hoc_latent_demand_paired_sensitivity",
        contract_version=CONTRACT_VERSION,
        training_window=f"{training_start.isoformat()} <= t < {training_end.isoformat()}",
        evaluation_window=f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}",
        stations=len(coordinates),
        seeds=seeds,
        scenarios=SCENARIO_ORDER,
        policies={
            "P0": {"description": "재배치 없음"},
            "P2-R": {
                "description": "동결 보호형 즉시출발 정책",
                "donor_reserve_bikes": 7,
                "max_actions_per_decision": 3,
                "average_speed_kmh": 20.0,
                "vehicle_capacity": 10,
                "road_distance_factor": 1.3,
                "handling_minutes_per_bike": 0.75,
            },
        },
        observed_baseline=results.observed_baseline,
        scenario_summary=tuple(results.scenario_summary.to_dicts()),
        station_robustness=results.station_robustness,
        stop_checks=results.stop_checks,
        stop_required=results.stop_required,
        interpretation_status=(
            "STOP_REQUIRED" if results.stop_required else "SENSITIVITY_COMPLETE"
        ),
        limitations=(
            "합성 요청은 관측 복원이 아니라 품절 검열시간 기반 Poisson 민감도다.",
            "평가 재고로 검열 위치를 표시하므로 독립 검증이 아닌 사후 분석이다.",
            "인근 대여소로 이동한 대체수요와 이용자의 포기·검색 행동을 모델링하지 않는다.",
            "P2-R은 실제 차량 차고지·교통·교대가 없는 공급지 즉시출발 실행모델이다.",
            "같은 5일 홀드아웃이 앞선 정책·fleet 분석에도 사용됐다.",
            "bootstrap 구간은 생성 seed 변동만 나타내며 모형 불확실성 전체를 포함하지 않는다.",
            "대여소 악화 확률은 운영상 서비스 안정성이며 인구·소득 형평성 지표가 아니다.",
            "80% persistent 기준은 설명용 분류이며 정책 선택이나 통계적 유의성 기준이 아니다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(experiment), encoding="utf-8")
    return experiment
