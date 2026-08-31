from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import polars as pl

CONTRACT_VERSION = "latent-demand-v2"
SCENARIO_ORDER = ("low", "base", "high")
SCENARIO_BOUNDARY_MINUTES = {"low": 15, "base": 30, "high": 45}


class LatentDemandError(RuntimeError):
    """잠재수요 입력이나 동결 계약을 만족하지 못할 때 발생하는 오류."""


@dataclass(frozen=True)
class LatentDemandConfig:
    training_start: datetime
    training_end: datetime
    evaluation_start: datetime
    evaluation_end: datetime
    seed: int = 20251124
    minimum_intensity_exposures: int = 3
    minimum_trip_pool_records: int = 20

    def __post_init__(self) -> None:
        if self.training_end <= self.training_start:
            raise ValueError("잠재수요 학습 종료 시각은 시작 시각보다 늦어야 합니다")
        if self.evaluation_end <= self.evaluation_start:
            raise ValueError("잠재수요 평가 종료 시각은 시작 시각보다 늦어야 합니다")
        if self.training_end > self.evaluation_start:
            raise ValueError("잠재수요 학습기간과 평가기간은 겹칠 수 없습니다")
        if self.minimum_intensity_exposures <= 0:
            raise ValueError("강도 pool 최소 노출시간은 양수여야 합니다")
        if self.minimum_trip_pool_records <= 0:
            raise ValueError("여행 pool 최소 표본수는 양수여야 합니다")


@dataclass(frozen=True)
class LatentDemandAudit:
    contract_version: str
    seed: int
    training_uncensored_intervals: int
    evaluation_censored_intervals: int
    evaluation_both_zero_intervals: int
    evaluation_boundary_zero_intervals: int
    intensity_rate_capped_intervals: int
    intensity_rate_capped_interval_rate: float
    generated_candidates_before_hard_cap: int
    dropped_candidates_by_hard_cap: int
    hard_capped_intervals: int
    hard_capped_interval_rate: float
    intensity_pool_intervals: dict[str, int]
    district_fallback_rate: float
    scenario_requests: dict[str, int]
    master_manifest_hash: str
    scenario_manifest_hashes: dict[str, str]


@dataclass(frozen=True)
class LatentDemandManifest:
    requests: pl.DataFrame
    audit: LatentDemandAudit

    def requests_for(self, scenario: str) -> pl.DataFrame:
        if scenario not in SCENARIO_ORDER:
            raise ValueError(f"알 수 없는 잠재수요 시나리오: {scenario}")
        allowed = SCENARIO_ORDER[: SCENARIO_ORDER.index(scenario) + 1]
        return self.requests.filter(pl.col("latent_min_scenario").is_in(allowed))


@dataclass(frozen=True)
class _StationInterval:
    station_id: str
    start: datetime
    start_bikes: int
    end_bikes: int
    rentals: int


@dataclass(frozen=True)
class _IntensityModel:
    raw_rate_per_hour: float
    rate_per_hour: float
    empirical_p95_per_hour: int
    pool_name: str
    training_exposures: int


@dataclass(frozen=True)
class _TripTemplate:
    destination_station_id: str
    duration_minutes: int
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class _TripPool:
    name: str
    templates: tuple[_TripTemplate, ...]


_REQUEST_SCHEMA = {
    "request_id": pl.String,
    "source": pl.String,
    "latent_min_scenario": pl.String,
    "rental_at": pl.Datetime("us"),
    "return_at": pl.Datetime("us"),
    "origin_station_id": pl.String,
    "destination_station_id": pl.String,
    "duration_minutes": pl.Int64,
    "generation_seed": pl.Int64,
    "censor_interval_id": pl.String,
    "intensity_pool": pl.String,
    "intensity_training_hours": pl.Int64,
    "trip_pool": pl.String,
    "trip_pool_records": pl.Int64,
    "raw_hourly_intensity": pl.Float64,
    "hourly_intensity": pl.Float64,
    "empirical_p95_per_hour": pl.Int64,
    "poisson_q99_high_window": pl.Int64,
    "hard_count_cap": pl.Int64,
    "contract_version": pl.String,
}


def build_latent_demand_manifest(
    trips: pl.DataFrame,
    station_hour: pl.DataFrame,
    config: LatentDemandConfig,
    *,
    eligible_station_ids: set[str] | None = None,
) -> LatentDemandManifest:
    """동결된 v2 계약에 따라 정책 독립적인 합성 요청 manifest를 만든다."""

    intervals = _build_station_intervals(station_hour, eligible_station_ids)
    training_intervals = [
        interval
        for interval in intervals
        if config.training_start <= interval.start < config.training_end
        and interval.start_bikes > 0
        and interval.end_bikes > 0
    ]
    if not training_intervals:
        raise LatentDemandError("학습기간에 양 끝 재고가 있는 비품절 60분 구간이 없습니다")

    intensity_pools = _build_intensity_pools(training_intervals)
    trip_pools = _build_trip_pools(trips, config)
    evaluation_intervals = [
        interval
        for interval in intervals
        if config.evaluation_start <= interval.start < config.evaluation_end
        and (interval.start_bikes == 0 or interval.end_bikes == 0)
    ]

    request_rows: list[dict[str, Any]] = []
    intensity_pool_intervals: defaultdict[str, int] = defaultdict(int)
    intensity_rate_capped_intervals = 0
    generated_before_hard_cap = 0
    dropped_by_hard_cap = 0
    hard_capped_intervals = 0
    both_zero_intervals = 0
    boundary_zero_intervals = 0

    for interval in evaluation_intervals:
        if interval.start_bikes == 0 and interval.end_bikes == 0:
            both_zero_intervals += 1
        else:
            boundary_zero_intervals += 1
        intensity = _select_intensity_model(interval, intensity_pools, config)
        intensity_pool_intervals[intensity.pool_name] += 1
        if intensity.rate_per_hour < intensity.raw_rate_per_hour:
            intensity_rate_capped_intervals += 1
        poisson_q99, hard_count_cap = _hard_count_cap(interval, intensity)
        candidates = _generate_high_candidates(interval, intensity, config.seed)
        generated_before_hard_cap += len(candidates)
        if len(candidates) > hard_count_cap:
            hard_capped_intervals += 1
            dropped_by_hard_cap += len(candidates) - hard_count_cap
            candidates = sorted(
                candidates,
                key=lambda candidate: _stable_int(
                    config.seed,
                    interval.station_id,
                    interval.start.isoformat(),
                    candidate[0],
                    "cap",
                ),
            )[:hard_count_cap]
            candidates.sort(key=lambda candidate: candidate[1])

        for ordinal, rental_at in candidates:
            request_id = (
                f"syn:{config.seed}:{interval.station_id}:"
                f"{interval.start:%Y%m%dT%H%M}:{ordinal:04d}"
            )
            trip_pool = _select_trip_pool(interval, trip_pools, config)
            template_index = _stable_int(config.seed, request_id, "trip") % len(trip_pool.templates)
            template = trip_pool.templates[template_index]
            request_rows.append(
                {
                    "request_id": request_id,
                    "source": "synthetic_latent",
                    "latent_min_scenario": _minimum_scenario(interval, rental_at),
                    "rental_at": rental_at,
                    "return_at": rental_at + timedelta(minutes=template.duration_minutes),
                    "origin_station_id": interval.station_id,
                    "destination_station_id": template.destination_station_id,
                    "duration_minutes": template.duration_minutes,
                    "generation_seed": config.seed,
                    "censor_interval_id": (f"{interval.station_id}:{interval.start:%Y%m%dT%H%M}"),
                    "intensity_pool": intensity.pool_name,
                    "intensity_training_hours": intensity.training_exposures,
                    "trip_pool": trip_pool.name,
                    "trip_pool_records": len(trip_pool.templates),
                    "raw_hourly_intensity": intensity.raw_rate_per_hour,
                    "hourly_intensity": intensity.rate_per_hour,
                    "empirical_p95_per_hour": intensity.empirical_p95_per_hour,
                    "poisson_q99_high_window": poisson_q99,
                    "hard_count_cap": hard_count_cap,
                    "contract_version": CONTRACT_VERSION,
                }
            )

    requests = pl.DataFrame(request_rows, schema=_REQUEST_SCHEMA).sort(
        "rental_at", "origin_station_id", "request_id"
    )
    scenario_frames = {
        scenario: _requests_for_scenario(requests, scenario) for scenario in SCENARIO_ORDER
    }
    scenario_counts = {scenario: frame.height for scenario, frame in scenario_frames.items()}
    if not (scenario_counts["low"] <= scenario_counts["base"] <= scenario_counts["high"]):
        raise LatentDemandError("잠재수요 시나리오 중첩 건수 불변조건을 위반했습니다")

    censored_count = len(evaluation_intervals)
    district_count = intensity_pool_intervals.get("district_day_hour", 0)
    audit = LatentDemandAudit(
        contract_version=CONTRACT_VERSION,
        seed=config.seed,
        training_uncensored_intervals=len(training_intervals),
        evaluation_censored_intervals=censored_count,
        evaluation_both_zero_intervals=both_zero_intervals,
        evaluation_boundary_zero_intervals=boundary_zero_intervals,
        intensity_rate_capped_intervals=intensity_rate_capped_intervals,
        intensity_rate_capped_interval_rate=_rate(intensity_rate_capped_intervals, censored_count),
        generated_candidates_before_hard_cap=generated_before_hard_cap,
        dropped_candidates_by_hard_cap=dropped_by_hard_cap,
        hard_capped_intervals=hard_capped_intervals,
        hard_capped_interval_rate=_rate(hard_capped_intervals, censored_count),
        intensity_pool_intervals=dict(sorted(intensity_pool_intervals.items())),
        district_fallback_rate=_rate(district_count, censored_count),
        scenario_requests=scenario_counts,
        master_manifest_hash=calculate_request_manifest_hash(requests),
        scenario_manifest_hashes={
            scenario: calculate_request_manifest_hash(frame)
            for scenario, frame in scenario_frames.items()
        },
    )
    return LatentDemandManifest(requests=requests, audit=audit)


def _build_station_intervals(
    station_hour: pl.DataFrame,
    eligible_station_ids: set[str] | None,
) -> list[_StationInterval]:
    required = {
        "timestamp",
        "station_id",
        "available_bikes",
        "rentals",
        "inventory_observed",
        "actionable",
    }
    missing = required - set(station_hour.columns)
    if missing:
        raise LatentDemandError(f"잠재수요 재고 필수 열 누락: {sorted(missing)}")

    frame = station_hour.filter(pl.col("actionable") & pl.col("inventory_observed"))
    if eligible_station_ids is not None:
        frame = frame.filter(pl.col("station_id").cast(pl.String).is_in(eligible_station_ids))
    frame = frame.select(
        pl.col("timestamp"),
        pl.col("station_id").cast(pl.String),
        pl.col("available_bikes"),
        pl.col("rentals"),
    ).sort("station_id", "timestamp")
    if frame.is_empty():
        raise LatentDemandError("잠재수요 분석 가능한 재고 행이 없습니다")
    duplicate_count = frame.group_by("station_id", "timestamp").len().filter(pl.col("len") > 1)
    if not duplicate_count.is_empty():
        raise LatentDemandError("잠재수요 재고 대여소-시각 키가 중복됐습니다")
    if frame["available_bikes"].null_count() or frame["rentals"].null_count():
        raise LatentDemandError("관측 재고 또는 대여 건수에 결측이 있습니다")
    if frame.filter((pl.col("available_bikes") < 0) | (pl.col("rentals") < 0)).height:
        raise LatentDemandError("재고와 대여 건수는 음수일 수 없습니다")

    by_station: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.iter_rows(named=True):
        by_station[str(row["station_id"])].append(row)
    intervals: list[_StationInterval] = []
    for station_id in sorted(by_station):
        station_rows = by_station[station_id]
        for start_row, end_row in zip(station_rows, station_rows[1:], strict=False):
            if end_row["timestamp"] - start_row["timestamp"] != timedelta(hours=1):
                continue
            intervals.append(
                _StationInterval(
                    station_id=station_id,
                    start=start_row["timestamp"],
                    start_bikes=int(start_row["available_bikes"]),
                    end_bikes=int(end_row["available_bikes"]),
                    rentals=int(start_row["rentals"]),
                )
            )
    return intervals


def _build_intensity_pools(
    intervals: list[_StationInterval],
) -> tuple[
    tuple[str, Callable[[_StationInterval], tuple[Any, ...]], dict[tuple[Any, ...], list[int]]], ...
]:
    definitions: tuple[tuple[str, Callable[[_StationInterval], tuple[Any, ...]]], ...] = (
        (
            "station_day_hour",
            lambda row: (row.station_id, _is_weekend(row.start), row.start.hour),
        ),
        (
            "station_day_block",
            lambda row: (row.station_id, _is_weekend(row.start), row.start.hour // 3),
        ),
        ("station_hour", lambda row: (row.station_id, row.start.hour)),
        ("station_block", lambda row: (row.station_id, row.start.hour // 3)),
        ("district_day_hour", lambda row: (_is_weekend(row.start), row.start.hour)),
    )
    pools = []
    for name, key_function in definitions:
        values: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
        for interval in intervals:
            values[key_function(interval)].append(interval.rentals)
        pools.append((name, key_function, dict(values)))
    return tuple(pools)


def _select_intensity_model(
    interval: _StationInterval,
    pools: tuple[
        tuple[str, Callable[[_StationInterval], tuple[Any, ...]], dict[tuple[Any, ...], list[int]]],
        ...,
    ],
    config: LatentDemandConfig,
) -> _IntensityModel:
    for name, key_function, values_by_key in pools:
        values = values_by_key.get(key_function(interval), [])
        sufficient = len(values) >= config.minimum_intensity_exposures
        if values and (sufficient or name == "district_day_hour"):
            raw_rate_per_hour = sum(values) / len(values)
            empirical_p95_per_hour = max(1, _nearest_rank_quantile(values, 0.95))
            return _IntensityModel(
                raw_rate_per_hour=raw_rate_per_hour,
                rate_per_hour=min(raw_rate_per_hour, empirical_p95_per_hour),
                empirical_p95_per_hour=empirical_p95_per_hour,
                pool_name=name,
                training_exposures=len(values),
            )
    raise LatentDemandError(
        f"잠재수요 강도 pool이 없습니다: {interval.station_id} {interval.start.isoformat()}"
    )


def _build_trip_pools(
    trips: pl.DataFrame,
    config: LatentDemandConfig,
) -> tuple[
    tuple[
        str, Callable[[str, datetime], tuple[Any, ...]], dict[tuple[Any, ...], list[_TripTemplate]]
    ],
    ...,
]:
    required = {
        "rent_at",
        "return_at",
        "rent_station_id",
        "return_station_id",
        "duration_minutes",
        "rent_in_scope",
    }
    missing = required - set(trips.columns)
    if missing:
        raise LatentDemandError(f"잠재수요 여행 필수 열 누락: {sorted(missing)}")
    frame = (
        trips.filter(
            (pl.col("rent_at") >= config.training_start)
            & (pl.col("rent_at") < config.training_end)
            & pl.col("rent_in_scope")
            & pl.col("return_at").is_not_null()
            & pl.col("return_station_id").is_not_null()
            & (pl.col("duration_minutes") > 0)
        )
        .select(
            pl.col("rent_at"),
            pl.col("rent_station_id").cast(pl.String),
            pl.col("return_station_id").cast(pl.String),
            pl.col("duration_minutes").cast(pl.Int64),
        )
        .sort("rent_at", "rent_station_id", "return_station_id", "duration_minutes")
    )
    definitions: tuple[tuple[str, Callable[[str, datetime], tuple[Any, ...]]], ...] = (
        ("origin_day_hour", lambda origin, at: (origin, _is_weekend(at), at.hour)),
        ("origin_day_block", lambda origin, at: (origin, _is_weekend(at), at.hour // 3)),
        ("origin_all", lambda origin, at: (origin,)),
        ("district_day_hour", lambda origin, at: (_is_weekend(at), at.hour)),
    )
    pools = []
    for name, key_function in definitions:
        templates: defaultdict[tuple[Any, ...], list[_TripTemplate]] = defaultdict(list)
        for row in frame.iter_rows(named=True):
            rent_at = row["rent_at"]
            origin = str(row["rent_station_id"])
            destination = str(row["return_station_id"])
            duration = int(row["duration_minutes"])
            template = _TripTemplate(
                destination_station_id=destination,
                duration_minutes=duration,
                sort_key=(rent_at, origin, destination, duration),
            )
            templates[key_function(origin, rent_at)].append(template)
        pools.append((name, key_function, dict(templates)))
    return tuple(pools)


def _select_trip_pool(
    interval: _StationInterval,
    pools: tuple[
        tuple[
            str,
            Callable[[str, datetime], tuple[Any, ...]],
            dict[tuple[Any, ...], list[_TripTemplate]],
        ],
        ...,
    ],
    config: LatentDemandConfig,
) -> _TripPool:
    for name, key_function, values_by_key in pools:
        values = values_by_key.get(key_function(interval.station_id, interval.start), [])
        sufficient = len(values) >= config.minimum_trip_pool_records
        if values and (sufficient or name == "district_day_hour"):
            ordered = tuple(sorted(values, key=lambda value: value.sort_key))
            return _TripPool(name=name, templates=ordered)
    raise LatentDemandError(
        f"잠재수요 OD·이용시간 pool이 없습니다: {interval.station_id} {interval.start.isoformat()}"
    )


def _generate_high_candidates(
    interval: _StationInterval,
    intensity: _IntensityModel,
    seed: int,
) -> list[tuple[int, datetime]]:
    if intensity.rate_per_hour <= 0:
        return []
    high_start, high_end = _high_window(interval)
    rng = random.Random(
        _stable_int(seed, interval.station_id, interval.start.isoformat(), "arrival")
    )
    rate_per_second = intensity.rate_per_hour / 3600
    elapsed_seconds = 0.0
    candidates: list[tuple[int, datetime]] = []
    while True:
        elapsed_seconds += rng.expovariate(rate_per_second)
        candidate_at = high_start + timedelta(seconds=elapsed_seconds)
        if candidate_at >= high_end:
            break
        candidates.append((len(candidates), candidate_at))
        if len(candidates) > 100_000:
            raise LatentDemandError("한 대여소-시간의 잠재수요 후보가 100,000건을 초과했습니다")
    return candidates


def _hard_count_cap(
    interval: _StationInterval,
    intensity: _IntensityModel,
) -> tuple[int, int]:
    high_start, high_end = _high_window(interval)
    high_exposure_hours = (high_end - high_start).total_seconds() / 3600
    expected_candidates = intensity.rate_per_hour * high_exposure_hours
    poisson_q99 = _poisson_quantile(expected_candidates, 0.99)
    return poisson_q99, max(intensity.empirical_p95_per_hour, poisson_q99)


def _high_window(interval: _StationInterval) -> tuple[datetime, datetime]:
    high_minutes = SCENARIO_BOUNDARY_MINUTES["high"]
    if interval.start_bikes == 0 and interval.end_bikes == 0:
        return interval.start, interval.start + timedelta(hours=1)
    if interval.start_bikes > 0 and interval.end_bikes == 0:
        return (
            interval.start + timedelta(minutes=60 - high_minutes),
            interval.start + timedelta(hours=1),
        )
    if interval.start_bikes == 0 and interval.end_bikes > 0:
        return interval.start, interval.start + timedelta(minutes=high_minutes)
    raise LatentDemandError("품절 증거가 없는 구간에서 잠재수요 창을 요청했습니다")


def _minimum_scenario(interval: _StationInterval, rental_at: datetime) -> str:
    low_minutes = SCENARIO_BOUNDARY_MINUTES["low"]
    base_minutes = SCENARIO_BOUNDARY_MINUTES["base"]
    if interval.start_bikes == 0 and interval.end_bikes == 0:
        return "low"
    if interval.start_bikes > 0 and interval.end_bikes == 0:
        if rental_at >= interval.start + timedelta(minutes=60 - low_minutes):
            return "low"
        if rental_at >= interval.start + timedelta(minutes=60 - base_minutes):
            return "base"
        return "high"
    if rental_at < interval.start + timedelta(minutes=low_minutes):
        return "low"
    if rental_at < interval.start + timedelta(minutes=base_minutes):
        return "base"
    return "high"


def _requests_for_scenario(requests: pl.DataFrame, scenario: str) -> pl.DataFrame:
    allowed = SCENARIO_ORDER[: SCENARIO_ORDER.index(scenario) + 1]
    return requests.filter(pl.col("latent_min_scenario").is_in(allowed))


def _nearest_rank_quantile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _poisson_quantile(mean: float, quantile: float) -> int:
    if mean < 0:
        raise ValueError("Poisson 평균은 음수일 수 없습니다")
    if not 0 < quantile < 1:
        raise ValueError("Poisson 분위수 확률은 0과 1 사이여야 합니다")
    if mean == 0:
        return 0
    if mean > 700:
        raise LatentDemandError("Poisson 평균이 안정적인 정확 계산 범위 700을 초과했습니다")
    probability = math.exp(-mean)
    cumulative = probability
    value = 0
    while cumulative < quantile:
        value += 1
        probability *= mean / value
        cumulative += probability
        if value > 100_000:
            raise LatentDemandError("Poisson 99백분위수 계산이 수렴하지 않았습니다")
    return value


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def calculate_request_manifest_hash(frame: pl.DataFrame) -> str:
    """Return the canonical SHA-256 hash used by latent-demand manifests."""

    if "request_id" not in frame.columns:
        raise LatentDemandError("manifest hash 계산에 request_id 열이 필요합니다")
    canonical_rows = []
    for row in frame.sort("request_id").iter_rows(named=True):
        canonical_rows.append(
            {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in row.items()
            }
        )
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_weekend(timestamp: datetime) -> bool:
    return timestamp.weekday() >= 5


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
