from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import matplotlib
import polars as pl
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class SimulationError(RuntimeError):
    """시뮬레이션 입력 또는 불변조건이 유효하지 않을 때 발생하는 오류."""


@dataclass(frozen=True)
class SimulationConfig:
    start: datetime
    end: datetime
    decision_interval_minutes: int = 60
    max_bikes_per_decision: int = 40

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("시뮬레이션 종료 시각은 시작 시각보다 늦어야 합니다")
        if self.decision_interval_minutes <= 0:
            raise ValueError("재배치 판단 주기는 양수여야 합니다")
        if self.max_bikes_per_decision <= 0:
            raise ValueError("판단 시점별 이동 한도는 양수여야 합니다")


@dataclass(frozen=True)
class Transfer:
    from_station_id: str
    to_station_id: str
    bike_count: int


class RelocationPolicy(Protocol):
    name: str

    def plan(
        self,
        inventory: dict[str, int],
        *,
        max_bikes: int,
    ) -> tuple[Transfer, ...]: ...


@dataclass(frozen=True)
class NoRelocationPolicy:
    name: str = "no_relocation"

    def plan(
        self,
        inventory: dict[str, int],
        *,
        max_bikes: int,
    ) -> tuple[Transfer, ...]:
        del inventory, max_bikes
        return ()


@dataclass(frozen=True)
class StaticThresholdPolicy:
    lower_threshold: int
    target_bikes: int
    upper_threshold: int
    name: str = "static_threshold"

    def __post_init__(self) -> None:
        if not 0 <= self.lower_threshold < self.target_bikes < self.upper_threshold:
            raise ValueError("재고 임계값은 0 <= 하한 < 목표 < 상한이어야 합니다")

    def plan(
        self,
        inventory: dict[str, int],
        *,
        max_bikes: int,
    ) -> tuple[Transfer, ...]:
        receivers = sorted(
            (
                [station_id, self.target_bikes - bikes]
                for station_id, bikes in inventory.items()
                if bikes < self.lower_threshold
            ),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        donors = sorted(
            (
                [station_id, bikes - self.target_bikes]
                for station_id, bikes in inventory.items()
                if bikes > self.upper_threshold
            ),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        transfers: list[Transfer] = []
        receiver_index = 0
        donor_index = 0
        remaining_budget = max_bikes
        while (
            receiver_index < len(receivers) and donor_index < len(donors) and remaining_budget > 0
        ):
            receiver_id, deficit = receivers[receiver_index]
            donor_id, surplus = donors[donor_index]
            bike_count = min(int(deficit), int(surplus), remaining_budget)
            if bike_count <= 0:
                break
            transfers.append(
                Transfer(
                    from_station_id=str(donor_id),
                    to_station_id=str(receiver_id),
                    bike_count=bike_count,
                )
            )
            receivers[receiver_index][1] = int(deficit) - bike_count
            donors[donor_index][1] = int(surplus) - bike_count
            remaining_budget -= bike_count
            if receivers[receiver_index][1] == 0:
                receiver_index += 1
            if donors[donor_index][1] == 0:
                donor_index += 1
        return tuple(transfers)


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: datetime
    priority: int
    sequence: int
    kind: str
    station_id: str = ""
    destination_id: str | None = None
    trip_id: int | None = None


@dataclass(frozen=True)
class ReplayScenario:
    config: SimulationConfig
    station_names: dict[str, str]
    initial_inventory: dict[str, int]
    events: tuple[ReplayEvent, ...]


@dataclass(frozen=True)
class SimulationMetrics:
    policy_name: str
    start: str
    end_exclusive: str
    stations: int
    observed_requests: int
    successful_rentals: int
    failed_rentals: int
    service_rate: float
    failures_per_1000_requests: float
    empty_station_minutes: float
    empty_station_hours: float
    empty_station_time_rate: float
    decision_epochs: int
    relocation_batches: int
    relocation_actions: int
    bikes_moved: int
    max_bikes_moved_in_epoch: int
    initial_station_bikes: int
    unconditional_inbound_returns: int
    successful_internal_returns: int
    outbound_departures: int
    in_transit_at_end: int
    final_station_bikes: int
    conservation_residual: int
    stations_with_requests: int
    p10_station_service_rate: float
    worst_station_id: str
    worst_station_name: str
    worst_station_service_rate: float


@dataclass(frozen=True)
class SimulationRun:
    metrics: SimulationMetrics
    station_metrics: pl.DataFrame


@dataclass(frozen=True)
class ThresholdCandidate:
    label: str
    lower_threshold: int
    target_bikes: int
    upper_threshold: int


@dataclass(frozen=True)
class SimulationExperiment:
    generated_at_utc: str
    method: str
    training_window: str
    evaluation_window: str
    decision_interval_minutes: int
    max_bikes_per_decision: int
    selected_candidate: dict[str, Any]
    training_runs: tuple[dict[str, Any], ...]
    evaluation_runs: tuple[dict[str, Any], ...]
    improvement: dict[str, float]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


DEFAULT_CANDIDATES = (
    ThresholdCandidate("conservative", 1, 4, 8),
    ThresholdCandidate("balanced", 2, 5, 8),
    ThresholdCandidate("service_first", 3, 6, 10),
)


def build_replay_scenario(
    trips: pl.DataFrame,
    station_hour: pl.DataFrame,
    config: SimulationConfig,
) -> ReplayScenario:
    required_trip_columns = {
        "rent_at",
        "return_at",
        "rent_station_id",
        "return_station_id",
    }
    required_station_columns = {
        "timestamp",
        "station_id",
        "station_name",
        "available_bikes",
        "inventory_observed",
        "actionable",
    }
    missing_trip_columns = required_trip_columns - set(trips.columns)
    missing_station_columns = required_station_columns - set(station_hour.columns)
    if missing_trip_columns:
        raise SimulationError(f"대여이력 필수 열 누락: {sorted(missing_trip_columns)}")
    if missing_station_columns:
        raise SimulationError(f"재고 필수 열 누락: {sorted(missing_station_columns)}")

    initial_rows = station_hour.filter(
        (pl.col("timestamp") == config.start) & pl.col("actionable") & pl.col("inventory_observed")
    ).select("station_id", "station_name", "available_bikes")
    if initial_rows.is_empty():
        raise SimulationError(f"시작 시각 재고가 없습니다: {config.start.isoformat()}")
    if initial_rows["station_id"].n_unique() != initial_rows.height:
        raise SimulationError("시작 시각 대여소 재고 키가 중복됐습니다")
    if initial_rows["available_bikes"].null_count() > 0:
        raise SimulationError("시작 시각에 결측 재고가 있습니다")

    initial_inventory = {
        str(row["station_id"]): int(row["available_bikes"])
        for row in initial_rows.iter_rows(named=True)
    }
    if any(value < 0 for value in initial_inventory.values()):
        raise SimulationError("시작 재고는 음수일 수 없습니다")
    station_names = {
        str(row["station_id"]): str(row["station_name"])
        for row in initial_rows.iter_rows(named=True)
    }
    station_ids = set(initial_inventory)

    relevant = (
        trips.with_row_index("trip_id")
        .filter(
            (
                (pl.col("rent_at") >= config.start)
                & (pl.col("rent_at") < config.end)
                & pl.col("rent_station_id").is_in(station_ids)
            )
            | (
                pl.col("return_at").is_not_null()
                & (pl.col("return_at") >= config.start)
                & (pl.col("return_at") < config.end)
                & pl.col("return_station_id").is_in(station_ids)
            )
        )
        .select(
            "trip_id",
            "rent_at",
            "return_at",
            "rent_station_id",
            "return_station_id",
        )
    )

    events: list[ReplayEvent] = []
    sequence = 0
    for row in relevant.iter_rows(named=True):
        trip_id = int(row["trip_id"])
        rent_at = row["rent_at"]
        return_at = row["return_at"]
        origin_id = str(row["rent_station_id"])
        destination_id = str(row["return_station_id"])
        origin_in_scope = origin_id in station_ids
        destination_in_scope = destination_id in station_ids
        rental_in_window = origin_in_scope and config.start <= rent_at < config.end
        conditional_return = (
            rental_in_window
            and destination_in_scope
            and return_at is not None
            and rent_at < return_at < config.end
        )

        if rental_in_window:
            events.append(
                ReplayEvent(
                    timestamp=rent_at,
                    priority=2,
                    sequence=sequence,
                    kind="rental",
                    station_id=origin_id,
                    destination_id=destination_id if destination_in_scope else None,
                    trip_id=trip_id,
                )
            )
            sequence += 1
        if conditional_return:
            events.append(
                ReplayEvent(
                    timestamp=return_at,
                    priority=0,
                    sequence=sequence,
                    kind="conditional_return",
                    station_id=destination_id,
                    trip_id=trip_id,
                )
            )
            sequence += 1
        elif (
            destination_in_scope
            and return_at is not None
            and config.start <= return_at < config.end
            and not rental_in_window
        ):
            events.append(
                ReplayEvent(
                    timestamp=return_at,
                    priority=0,
                    sequence=sequence,
                    kind="unconditional_return",
                    station_id=destination_id,
                    trip_id=trip_id,
                )
            )
            sequence += 1

    decision_at = config.start
    while decision_at < config.end:
        events.append(
            ReplayEvent(
                timestamp=decision_at,
                priority=1,
                sequence=sequence,
                kind="decision",
            )
        )
        sequence += 1
        decision_at += timedelta(minutes=config.decision_interval_minutes)
    events.sort(key=lambda event: (event.timestamp, event.priority, event.sequence))
    return ReplayScenario(
        config=config,
        station_names=station_names,
        initial_inventory=initial_inventory,
        events=tuple(events),
    )


def simulate_replay(scenario: ReplayScenario, policy: RelocationPolicy) -> SimulationRun:
    config = scenario.config
    inventory = dict(scenario.initial_inventory)
    empty_since: dict[str, datetime | None] = {
        station_id: config.start if bikes == 0 else None for station_id, bikes in inventory.items()
    }
    empty_minutes = dict.fromkeys(inventory, 0.0)
    requests = dict.fromkeys(inventory, 0)
    successes = dict.fromkeys(inventory, 0)
    relocated_in = dict.fromkeys(inventory, 0)
    relocated_out = dict.fromkeys(inventory, 0)
    active_internal_trips: set[int] = set()
    observed_requests = 0
    successful_rentals = 0
    failed_rentals = 0
    decision_epochs = 0
    relocation_batches = 0
    relocation_actions = 0
    bikes_moved = 0
    max_bikes_moved_in_epoch = 0
    unconditional_returns = 0
    successful_internal_returns = 0
    outbound_departures = 0

    def change_inventory(station_id: str, delta: int, timestamp: datetime) -> None:
        before = inventory[station_id]
        after = before + delta
        if after < 0:
            raise SimulationError(f"음수 재고 발생: {station_id} {timestamp.isoformat()}")
        inventory[station_id] = after
        if before == 0 and after > 0:
            started_at = empty_since[station_id]
            if started_at is None:
                raise SimulationError("빈 재고 시작 시각이 누락됐습니다")
            empty_minutes[station_id] += (timestamp - started_at).total_seconds() / 60
            empty_since[station_id] = None
        elif before > 0 and after == 0:
            empty_since[station_id] = timestamp

    for event in scenario.events:
        if event.kind == "unconditional_return":
            change_inventory(event.station_id, 1, event.timestamp)
            unconditional_returns += 1
        elif event.kind == "conditional_return":
            if event.trip_id in active_internal_trips:
                change_inventory(event.station_id, 1, event.timestamp)
                active_internal_trips.remove(event.trip_id)
                successful_internal_returns += 1
        elif event.kind == "decision":
            decision_epochs += 1
            transfers = policy.plan(
                dict(inventory),
                max_bikes=config.max_bikes_per_decision,
            )
            moved_this_epoch = sum(transfer.bike_count for transfer in transfers)
            if moved_this_epoch > config.max_bikes_per_decision:
                raise SimulationError("정책이 판단 시점별 재배치 한도를 초과했습니다")
            if transfers:
                relocation_batches += 1
            for transfer in transfers:
                if transfer.bike_count <= 0:
                    raise SimulationError("재배치 대수는 양수여야 합니다")
                if transfer.from_station_id == transfer.to_station_id:
                    raise SimulationError("동일 대여소 간 재배치는 허용되지 않습니다")
                if (
                    transfer.from_station_id not in inventory
                    or transfer.to_station_id not in inventory
                ):
                    raise SimulationError("운영 대상 밖 대여소가 재배치에 포함됐습니다")
                change_inventory(
                    transfer.from_station_id,
                    -transfer.bike_count,
                    event.timestamp,
                )
                change_inventory(
                    transfer.to_station_id,
                    transfer.bike_count,
                    event.timestamp,
                )
                relocated_out[transfer.from_station_id] += transfer.bike_count
                relocated_in[transfer.to_station_id] += transfer.bike_count
                relocation_actions += 1
                bikes_moved += transfer.bike_count
            max_bikes_moved_in_epoch = max(max_bikes_moved_in_epoch, moved_this_epoch)
        elif event.kind == "rental":
            observed_requests += 1
            requests[event.station_id] += 1
            if inventory[event.station_id] == 0:
                failed_rentals += 1
                continue
            change_inventory(event.station_id, -1, event.timestamp)
            successful_rentals += 1
            successes[event.station_id] += 1
            if event.destination_id is None:
                outbound_departures += 1
            elif event.trip_id is not None:
                active_internal_trips.add(event.trip_id)
        else:
            raise SimulationError(f"알 수 없는 이벤트: {event.kind}")

    for station_id, started_at in empty_since.items():
        if started_at is not None:
            empty_minutes[station_id] += (config.end - started_at).total_seconds() / 60

    station_records = []
    for station_id in sorted(inventory):
        station_requests = requests[station_id]
        station_successes = successes[station_id]
        station_records.append(
            {
                "policy_name": policy.name,
                "station_id": station_id,
                "station_name": scenario.station_names[station_id],
                "requests": station_requests,
                "successful_rentals": station_successes,
                "failed_rentals": station_requests - station_successes,
                "service_rate": _rate(station_successes, station_requests),
                "empty_minutes": round(empty_minutes[station_id], 3),
                "empty_rate": _rate(
                    empty_minutes[station_id],
                    (config.end - config.start).total_seconds() / 60,
                ),
                "relocated_in": relocated_in[station_id],
                "relocated_out": relocated_out[station_id],
                "initial_bikes": scenario.initial_inventory[station_id],
                "final_bikes": inventory[station_id],
            }
        )
    station_metrics = pl.DataFrame(station_records)
    eligible_service = station_metrics.filter(pl.col("requests") >= 10).sort("service_rate")
    if eligible_service.is_empty():
        eligible_service = station_metrics.filter(pl.col("requests") > 0).sort("service_rate")
    worst = eligible_service.row(0, named=True)
    service_rates = eligible_service["service_rate"].sort().to_list()
    p10_index = min(len(service_rates) - 1, int(len(service_rates) * 0.1))
    total_empty_minutes = sum(empty_minutes.values())
    initial_bikes = sum(scenario.initial_inventory.values())
    final_bikes = sum(inventory.values())
    conservation_residual = (
        initial_bikes
        + unconditional_returns
        - outbound_departures
        - len(active_internal_trips)
        - final_bikes
    )
    if conservation_residual != 0:
        raise SimulationError(f"자전거 보존식 위반: residual={conservation_residual}")
    if successful_rentals + failed_rentals != observed_requests:
        raise SimulationError("대여 성공·실패 합이 전체 요청과 다릅니다")

    period_minutes = (config.end - config.start).total_seconds() / 60
    metrics = SimulationMetrics(
        policy_name=policy.name,
        start=config.start.isoformat(),
        end_exclusive=config.end.isoformat(),
        stations=len(inventory),
        observed_requests=observed_requests,
        successful_rentals=successful_rentals,
        failed_rentals=failed_rentals,
        service_rate=_rate(successful_rentals, observed_requests),
        failures_per_1000_requests=round(1_000 * _rate(failed_rentals, observed_requests), 3),
        empty_station_minutes=round(total_empty_minutes, 3),
        empty_station_hours=round(total_empty_minutes / 60, 3),
        empty_station_time_rate=_rate(total_empty_minutes, len(inventory) * period_minutes),
        decision_epochs=decision_epochs,
        relocation_batches=relocation_batches,
        relocation_actions=relocation_actions,
        bikes_moved=bikes_moved,
        max_bikes_moved_in_epoch=max_bikes_moved_in_epoch,
        initial_station_bikes=initial_bikes,
        unconditional_inbound_returns=unconditional_returns,
        successful_internal_returns=successful_internal_returns,
        outbound_departures=outbound_departures,
        in_transit_at_end=len(active_internal_trips),
        final_station_bikes=final_bikes,
        conservation_residual=conservation_residual,
        stations_with_requests=station_metrics.filter(pl.col("requests") > 0).height,
        p10_station_service_rate=float(service_rates[p10_index]),
        worst_station_id=str(worst["station_id"]),
        worst_station_name=str(worst["station_name"]),
        worst_station_service_rate=float(worst["service_rate"]),
    )
    return SimulationRun(metrics=metrics, station_metrics=station_metrics)


def build_policy_comparison(
    *,
    trips_path: Path,
    station_hour_path: Path,
    training_start: datetime,
    training_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
    decision_interval_minutes: int,
    max_bikes_per_decision: int,
    training_csv_path: Path,
    comparison_csv_path: Path,
    station_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> SimulationExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"시뮬레이션 입력 Parquet을 읽지 못했습니다: {exc}") from exc

    training_config = SimulationConfig(
        start=training_start,
        end=training_end,
        decision_interval_minutes=decision_interval_minutes,
        max_bikes_per_decision=max_bikes_per_decision,
    )
    evaluation_config = SimulationConfig(
        start=evaluation_start,
        end=evaluation_end,
        decision_interval_minutes=decision_interval_minutes,
        max_bikes_per_decision=max_bikes_per_decision,
    )
    training_scenario = build_replay_scenario(trips, station_hour, training_config)
    baseline_training = simulate_replay(training_scenario, NoRelocationPolicy())
    candidate_training_runs: list[tuple[ThresholdCandidate, SimulationRun]] = []
    for candidate in DEFAULT_CANDIDATES:
        run = simulate_replay(
            training_scenario,
            StaticThresholdPolicy(
                lower_threshold=candidate.lower_threshold,
                target_bikes=candidate.target_bikes,
                upper_threshold=candidate.upper_threshold,
                name=f"static_threshold_{candidate.label}",
            ),
        )
        candidate_training_runs.append((candidate, run))
    selected_candidate, selected_training = max(
        candidate_training_runs,
        key=lambda item: (
            item[1].metrics.successful_rentals,
            -item[1].metrics.bikes_moved,
            -item[1].metrics.empty_station_minutes,
        ),
    )

    evaluation_scenario = build_replay_scenario(trips, station_hour, evaluation_config)
    baseline_evaluation = simulate_replay(evaluation_scenario, NoRelocationPolicy())
    threshold_evaluation = simulate_replay(
        evaluation_scenario,
        StaticThresholdPolicy(
            lower_threshold=selected_candidate.lower_threshold,
            target_bikes=selected_candidate.target_bikes,
            upper_threshold=selected_candidate.upper_threshold,
            name="static_threshold",
        ),
    )

    training_records = [
        {
            "candidate": "no_relocation",
            "lower_threshold": None,
            "target_bikes": None,
            "upper_threshold": None,
            **asdict(baseline_training.metrics),
        }
    ]
    training_records.extend(
        {
            "candidate": candidate.label,
            "lower_threshold": candidate.lower_threshold,
            "target_bikes": candidate.target_bikes,
            "upper_threshold": candidate.upper_threshold,
            **asdict(run.metrics),
        }
        for candidate, run in candidate_training_runs
    )
    evaluation_records = [
        {"policy_label": "P0 재배치 없음", **asdict(baseline_evaluation.metrics)},
        {"policy_label": "P1 고정 임계값", **asdict(threshold_evaluation.metrics)},
    ]
    training_frame = pl.DataFrame(training_records)
    comparison_frame = pl.DataFrame(evaluation_records)
    station_frame = pl.concat(
        [baseline_evaluation.station_metrics, threshold_evaluation.station_metrics],
        how="vertical",
    )
    training_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    station_csv_path.parent.mkdir(parents=True, exist_ok=True)
    training_frame.write_csv(training_csv_path)
    comparison_frame.write_csv(comparison_csv_path)
    station_frame.write_csv(station_csv_path)
    _plot_comparison(comparison_frame, figure_path)

    baseline_metrics = baseline_evaluation.metrics
    threshold_metrics = threshold_evaluation.metrics
    failures_avoided = baseline_metrics.failed_rentals - threshold_metrics.failed_rentals
    empty_hours_reduced = (
        baseline_metrics.empty_station_hours - threshold_metrics.empty_station_hours
    )
    improvement = {
        "failures_avoided": float(failures_avoided),
        "failure_reduction_rate": _rate(failures_avoided, baseline_metrics.failed_rentals),
        "service_rate_percentage_point": round(
            (threshold_metrics.service_rate - baseline_metrics.service_rate) * 100,
            3,
        ),
        "empty_station_hours_reduced": round(empty_hours_reduced, 3),
        "empty_time_reduction_rate": _rate(
            empty_hours_reduced,
            baseline_metrics.empty_station_hours,
        ),
        "failures_avoided_per_100_bikes_moved": round(
            100 * _rate(failures_avoided, threshold_metrics.bikes_moved),
            3,
        ),
    }
    output_files = {
        "training_grid": str(training_csv_path),
        "policy_comparison": str(comparison_csv_path),
        "station_comparison": str(station_csv_path),
        "comparison_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = SimulationExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="observed_successful_trip_deterministic_replay",
        training_window=f"{training_start.isoformat()} <= t < {training_end.isoformat()}",
        evaluation_window=f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}",
        decision_interval_minutes=decision_interval_minutes,
        max_bikes_per_decision=max_bikes_per_decision,
        selected_candidate={
            **asdict(selected_candidate),
            "training_successful_rentals": selected_training.metrics.successful_rentals,
            "selection_rule": "successful_rentals desc, bikes_moved asc, empty_minutes asc",
        },
        training_runs=tuple(training_records),
        evaluation_runs=tuple(evaluation_records),
        improvement=improvement,
        limitations=(
            "공개 대여이력에 기록된 성공 요청만 재생하며 미관측 잠재 수요는 포함하지 않는다.",
            "실제 운영 재배치 이력이 없어 무재배치는 운영개입을 제거한 스트레스 테스트다.",
            "명목 거치대 수를 하드 용량으로 사용하지 않아 반납 실패는 계산하지 않는다.",
            "P1 이동은 판단 시점에 즉시 완료되며 이동거리와 차량 경로는 아직 반영하지 않는다.",
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


def _render_markdown(experiment: SimulationExperiment) -> str:
    training_rows = "\n".join(
        "| {candidate} | {lower} | {target} | {upper} | {service:.2%} | {failed:,} | "
        "{empty:,.1f} | {moved:,} |".format(
            candidate=row["candidate"],
            lower=row["lower_threshold"] if row["lower_threshold"] is not None else "-",
            target=row["target_bikes"] if row["target_bikes"] is not None else "-",
            upper=row["upper_threshold"] if row["upper_threshold"] is not None else "-",
            service=row["service_rate"],
            failed=row["failed_rentals"],
            empty=row["empty_station_hours"],
            moved=row["bikes_moved"],
        )
        for row in experiment.training_runs
    )
    evaluation_rows = "\n".join(
        "| {label} | {requests:,} | {service:.2%} | {failed:,} | {per1000:.1f} | "
        "{empty:,.1f} | {empty_rate:.2%} | {moved:,} | {batches:,} |".format(
            label=row["policy_label"],
            requests=row["observed_requests"],
            service=row["service_rate"],
            failed=row["failed_rentals"],
            per1000=row["failures_per_1000_requests"],
            empty=row["empty_station_hours"],
            empty_rate=row["empty_station_time_rate"],
            moved=row["bikes_moved"],
            batches=row["relocation_batches"],
        )
        for row in experiment.evaluation_runs
    )
    fairness_rows = "\n".join(
        "| {label} | {p10:.2%} | {worst_id} {worst_name} | {worst_rate:.2%} |".format(
            label=row["policy_label"],
            p10=row["p10_station_service_rate"],
            worst_id=row["worst_station_id"],
            worst_name=row["worst_station_name"],
            worst_rate=row["worst_station_service_rate"],
        )
        for row in experiment.evaluation_runs
    )
    selected = experiment.selected_candidate
    improvement = experiment.improvement
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    return f"""# 강남구 재배치 정책 시뮬레이션 비교

## 결론

- 검증기간 관측 요청 중 방지한 실패: {improvement["failures_avoided"]:,.0f}건
- 대여 성공률 개선: {improvement["service_rate_percentage_point"]:.3f}%p
- 빈 대여소 시간 감소: {improvement["empty_station_hours_reduced"]:,.1f}시간 \
({improvement["empty_time_reduction_rate"]:.2%})
- 재배치 100대당 방지한 실패: \
{improvement["failures_avoided_per_100_bikes_moved"]:.2f}건

이 결과는 실제 정책의 인과효과가 아니라, 동일한 관측 요청과 초기 재고에서 운영 개입을 \
바꾼 결정론적 스트레스 테스트다. 이동시간을 제외했으므로 P1 효과의 낙관적 상한에 가깝다.

## 실험 계약

- 방법: `{experiment.method}`
- 학습 구간: `{experiment.training_window}`
- 검증 구간: `{experiment.evaluation_window}`
- 운영 대상: 평가 시작 시 재고가 관측된 분석 대상 대여소
- 판단 주기: {experiment.decision_interval_minutes}분
- 판단 시점별 최대 이동: {experiment.max_bikes_per_decision}대
- 이벤트 우선순위: 반납 → 재배치 → 대여
- 선택된 P1: `{selected["label"]}` \
(하한 {selected["lower_threshold"]}, 목표 {selected["target_bikes"]}, \
상한 {selected["upper_threshold"]})
- 선택 규칙: 학습 성공건수 최대 → 동률이면 이동 대수 최소 → 빈 시간 최소

## 학습 구간 후보 비교

| 후보 | 하한 | 목표 | 상한 | 성공률 | 실패 | 빈 시간(대여소·시간) | 이동 대수 |
|---|---:|---:|---:|---:|---:|---:|---:|
{training_rows}

## 홀드아웃 검증 결과

| 정책 | 관측 요청 | 성공률 | 실패 | 1,000건당 실패 | 빈 시간 | 빈 시간률 | 이동 대수 | 실행 배치 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{evaluation_rows}

## 대여소별 서비스 형평성

대여 요청이 10건 이상인 대여소만 사용해 하위 10%와 최저 성공률을 비교했다.

| 정책 | 대여소 성공률 p10 | 최저 대여소 | 최저 성공률 |
|---|---:|---|---:|
{fairness_rows}

## 불변조건 검증

- 두 정책은 같은 초기 재고와 같은 요청 순서를 사용한다.
- 성공 요청 + 실패 요청 = 전체 관측 요청이다.
- 대여소 재고는 음수가 되지 않는다.
- 재배치는 판단 시점별 {experiment.max_bikes_per_decision}대를 넘지 않는다.
- 각 실행의 자전거 보존식 잔차는 0이다.

## 해석 제한

{limitations}

## 다음 단계

1. 대여소 좌표를 고정해 이동거리와 차량 이동시간을 반영한다.
2. 가까운 공급 대여소를 고르는 P2 greedy-nearest를 추가한다.
3. 잠재 수요를 낮음·기준·높음 시나리오로 생성해 민감도를 비교한다.
"""


def _plot_comparison(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = frame["policy_label"].to_list()
    panels = (
        ("failures_per_1000_requests", "1,000건당 대여 실패", "건"),
        ("empty_station_hours", "빈 대여소 누적 시간", "대여소·시간"),
        ("bikes_moved", "재배치 작업량", "대"),
    )
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    with plt.rc_context({"font.family": font_family}):
        figure, axes = plt.subplots(1, 3, figsize=(14, 5.2))
        for axis, (column, title, unit) in zip(axes, panels, strict=True):
            values = frame[column].to_list()
            bars = axis.bar(labels, values, color=["#7f7f7f", "#1f77b4"])
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.2)
            axis.tick_params(axis="x", rotation=10)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:,.1f}",
                    ha="center",
                    va="bottom",
                )
        figure.suptitle("강남구 2025-11 홀드아웃: 재배치 정책 비교")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def _rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
