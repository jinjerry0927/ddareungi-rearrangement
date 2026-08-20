from __future__ import annotations

import heapq
import json
import math
import re
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import matplotlib
import polars as pl
from matplotlib import font_manager

from ddareungi_rearrangement.seoul_api import LiveBikePage

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
    distance_km: float = 0.0
    travel_minutes: float = 0.0


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
class GreedyNearestPolicy:
    coordinates: dict[str, tuple[float, float]]
    lower_threshold: int = 2
    target_bikes: int = 5
    upper_threshold: int = 8
    max_actions_per_decision: int = 2
    vehicle_capacity: int = 20
    average_speed_kmh: float = 15.0
    road_distance_factor: float = 1.3
    handling_minutes_per_bike: float = 0.75
    name: str = "greedy_nearest"

    def __post_init__(self) -> None:
        if not 0 <= self.lower_threshold < self.target_bikes < self.upper_threshold:
            raise ValueError("재고 임계값은 0 <= 하한 < 목표 < 상한이어야 합니다")
        if self.max_actions_per_decision <= 0 or self.vehicle_capacity <= 0:
            raise ValueError("작업 수와 차량 용량은 양수여야 합니다")
        if self.average_speed_kmh <= 0 or self.road_distance_factor < 1:
            raise ValueError("차량 속도는 양수이고 거리 보정계수는 1 이상이어야 합니다")
        if self.handling_minutes_per_bike < 0:
            raise ValueError("상하차 시간은 음수일 수 없습니다")

    def plan(
        self,
        inventory: dict[str, int],
        *,
        max_bikes: int,
    ) -> tuple[Transfer, ...]:
        missing_coordinates = set(inventory) - set(self.coordinates)
        if missing_coordinates:
            raise SimulationError(f"P2 좌표 누락: {', '.join(sorted(missing_coordinates)[:5])}")
        receivers = sorted(
            (
                [station_id, self.target_bikes - bikes]
                for station_id, bikes in inventory.items()
                if bikes < self.lower_threshold
            ),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        donor_surplus = {
            station_id: bikes - self.target_bikes
            for station_id, bikes in inventory.items()
            if bikes > self.upper_threshold
        }
        transfers: list[Transfer] = []
        remaining_budget = max_bikes
        for receiver_id, raw_deficit in receivers:
            deficit = int(raw_deficit)
            while (
                deficit > 0
                and donor_surplus
                and remaining_budget > 0
                and len(transfers) < self.max_actions_per_decision
            ):
                donor_id = min(
                    donor_surplus,
                    key=lambda station_id: (
                        _haversine_km(
                            self.coordinates[str(station_id)],
                            self.coordinates[str(receiver_id)],
                        ),
                        str(station_id),
                    ),
                )
                bike_count = min(
                    deficit,
                    donor_surplus[donor_id],
                    self.vehicle_capacity,
                    remaining_budget,
                )
                straight_distance = _haversine_km(
                    self.coordinates[str(donor_id)],
                    self.coordinates[str(receiver_id)],
                )
                distance_km = straight_distance * self.road_distance_factor
                travel_minutes = (
                    distance_km / self.average_speed_kmh * 60
                    + bike_count * self.handling_minutes_per_bike
                )
                transfers.append(
                    Transfer(
                        from_station_id=str(donor_id),
                        to_station_id=str(receiver_id),
                        bike_count=bike_count,
                        distance_km=round(distance_km, 6),
                        travel_minutes=round(travel_minutes, 3),
                    )
                )
                deficit -= bike_count
                remaining_budget -= bike_count
                donor_surplus[donor_id] -= bike_count
                if donor_surplus[donor_id] == 0:
                    del donor_surplus[donor_id]
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
    bike_count: int = 0


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
    max_relocation_actions_in_epoch: int
    relocation_distance_km: float
    relocation_vehicle_minutes: float
    relocation_bikes_in_transit_at_end: int
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
    event_trace: pl.DataFrame | None = None


@dataclass(frozen=True)
class ThresholdCandidate:
    label: str
    lower_threshold: int
    target_bikes: int
    upper_threshold: int


@dataclass(frozen=True)
class CoordinateSnapshot:
    captured_at_utc: str
    source_rows: int
    requested_stations: int
    matched_stations: int
    coverage_rate: float
    missing_station_ids: tuple[str, ...]
    output_file: str


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


@dataclass(frozen=True)
class SpatialSimulationExperiment:
    generated_at_utc: str
    method: str
    evaluation_window: str
    stations: int
    coordinate_file: str
    excluded_station_ids: tuple[str, ...]
    assumptions: dict[str, Any]
    evaluation_runs: tuple[dict[str, Any], ...]
    comparisons: dict[str, dict[str, Any]]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


@dataclass(frozen=True)
class SensitivityExperiment:
    generated_at_utc: str
    method: str
    evaluation_window: str
    stations: int
    coordinate_file: str
    excluded_station_ids: tuple[str, ...]
    factors: dict[str, Any]
    baseline: dict[str, Any]
    runs: tuple[dict[str, Any], ...]
    service_best: dict[str, Any]
    empty_time_best: dict[str, Any]
    distance_efficiency_best: dict[str, Any]
    default_scenario: dict[str, Any]
    factor_findings: dict[str, Any]
    marginal_actions: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


@dataclass(frozen=True)
class TemporalRobustnessExperiment:
    generated_at_utc: str
    method: str
    analysis_window: str
    stations: int
    coordinate_file: str
    valid_days: int
    excluded_days: tuple[dict[str, Any], ...]
    policies: dict[str, dict[str, Any]]
    daily_runs: tuple[dict[str, Any], ...]
    group_summaries: tuple[dict[str, Any], ...]
    effect_consistency: dict[str, dict[str, Any]]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


@dataclass(frozen=True)
class StationEquityExperiment:
    generated_at_utc: str
    method: str
    evaluation_window: str
    stations: int
    coordinate_file: str
    policies: dict[str, dict[str, Any]]
    policy_runs: tuple[dict[str, Any], ...]
    equity_summaries: dict[str, dict[str, Any]]
    station_results: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


@dataclass(frozen=True)
class HarmTraceExperiment:
    generated_at_utc: str
    method: str
    evaluation_window: str
    stations: int
    policies: dict[str, dict[str, Any]]
    policy_runs: tuple[dict[str, Any], ...]
    transition_summaries: dict[str, dict[str, Any]]
    harm_requests: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    output_files: dict[str, str]


DEFAULT_CANDIDATES = (
    ThresholdCandidate("conservative", 1, 4, 8),
    ThresholdCandidate("balanced", 2, 5, 8),
    ThresholdCandidate("service_first", 3, 6, 10),
)


def snapshot_actionable_coordinates(
    pages: list[LiveBikePage],
    *,
    station_hour_path: Path,
    output_path: Path,
    captured_at_utc: datetime | None = None,
) -> CoordinateSnapshot:
    try:
        station_hour = pl.read_parquet(station_hour_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"대여소 시간 Parquet을 읽지 못했습니다: {exc}") from exc
    required = {"station_id", "station_name", "actionable"}
    missing_columns = required - set(station_hour.columns)
    if missing_columns:
        raise SimulationError(f"좌표 매핑 필수 열 누락: {sorted(missing_columns)}")

    actionable = (
        station_hour.filter(pl.col("actionable"))
        .select("station_id", "station_name")
        .unique(subset=["station_id"])
        .sort("station_id")
    )
    requested = {
        str(row["station_id"]): str(row["station_name"]) for row in actionable.iter_rows(named=True)
    }
    live_by_display_id: dict[str, dict[str, Any]] = {}
    source_rows = 0
    for page in pages:
        for row in page.rows:
            source_rows += 1
            live_name = str(row.get("stationName", "")).strip()
            match = re.match(r"^\s*(\d+)\s*\.\s*(.+)$", live_name)
            if not match:
                continue
            display_id = match.group(1)
            try:
                latitude = float(row.get("stationLatitude", ""))
                longitude = float(row.get("stationLongitude", ""))
            except (TypeError, ValueError):
                continue
            if not 37.0 <= latitude <= 38.0 or not 126.0 <= longitude <= 128.0:
                continue
            candidate = {
                "live_station_id": str(row.get("stationId", "")).strip(),
                "station_name_live": match.group(2).strip(),
                "latitude": latitude,
                "longitude": longitude,
            }
            existing = live_by_display_id.get(display_id)
            if existing is not None and existing != candidate:
                raise SimulationError(f"실시간 좌표 ID 중복 충돌: {display_id}")
            live_by_display_id[display_id] = candidate

    captured_at = captured_at_utc or datetime.now(UTC)
    records = []
    for station_id in sorted(requested, key=lambda value: int(value)):
        live = live_by_display_id.get(station_id)
        if live is None:
            continue
        records.append(
            {
                "station_id": station_id,
                "station_name_2025": requested[station_id],
                **live,
                "captured_at_utc": captured_at.isoformat(),
                "source": "Seoul Open Data live bike API",
            }
        )
    if not records:
        raise SimulationError("분석 대상 대여소 좌표를 하나도 매핑하지 못했습니다")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_csv(output_path)
    matched_ids = {str(record["station_id"]) for record in records}
    missing_ids = tuple(sorted(set(requested) - matched_ids, key=lambda value: int(value)))
    return CoordinateSnapshot(
        captured_at_utc=captured_at.isoformat(),
        source_rows=source_rows,
        requested_stations=len(requested),
        matched_stations=len(records),
        coverage_rate=_rate(len(records), len(requested)),
        missing_station_ids=missing_ids,
        output_file=str(output_path),
    )


def build_replay_scenario(
    trips: pl.DataFrame,
    station_hour: pl.DataFrame,
    config: SimulationConfig,
    *,
    eligible_station_ids: set[str] | None = None,
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

    initial_filter = (
        (pl.col("timestamp") == config.start) & pl.col("actionable") & pl.col("inventory_observed")
    )
    if eligible_station_ids is not None:
        initial_filter &= pl.col("station_id").is_in(eligible_station_ids)
    initial_rows = station_hour.filter(initial_filter).select(
        "station_id", "station_name", "available_bikes"
    )
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
    if eligible_station_ids is not None and station_ids != eligible_station_ids:
        unavailable = eligible_station_ids - station_ids
        raise SimulationError(
            f"좌표 대상 중 시작 재고가 없는 대여소: {', '.join(sorted(unavailable)[:5]) or 'none'}"
        )

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


def simulate_replay(
    scenario: ReplayScenario,
    policy: RelocationPolicy,
    *,
    collect_trace: bool = False,
) -> SimulationRun:
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
    max_relocation_actions_in_epoch = 0
    relocation_distance_km = 0.0
    relocation_vehicle_minutes = 0.0
    relocation_bikes_in_transit = 0
    unconditional_returns = 0
    successful_internal_returns = 0
    outbound_departures = 0
    trace_records: list[dict[str, Any]] = []
    trace_sequence = 0

    def record_trace(
        *,
        timestamp: datetime,
        event_kind: str,
        station_id: str,
        counterparty_station_id: str | None = None,
        trip_id: int | None = None,
        bike_count: int = 0,
        inventory_before: int,
        inventory_after: int,
        rental_outcome: str | None = None,
        distance_km: float = 0.0,
        travel_minutes: float = 0.0,
    ) -> None:
        nonlocal trace_sequence
        if not collect_trace:
            return
        trace_records.append(
            {
                "policy_name": policy.name,
                "timestamp": timestamp,
                "trace_sequence": trace_sequence,
                "event_kind": event_kind,
                "station_id": station_id,
                "counterparty_station_id": counterparty_station_id,
                "trip_id": trip_id,
                "bike_count": bike_count,
                "inventory_before": inventory_before,
                "inventory_after": inventory_after,
                "rental_outcome": rental_outcome,
                "distance_km": distance_km,
                "travel_minutes": travel_minutes,
            }
        )
        trace_sequence += 1

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

    event_queue = [
        (event.timestamp, event.priority, event.sequence, event) for event in scenario.events
    ]
    heapq.heapify(event_queue)
    next_sequence = max((event.sequence for event in scenario.events), default=-1) + 1
    while event_queue:
        _, _, _, event = heapq.heappop(event_queue)
        if event.timestamp >= config.end:
            break
        if event.kind == "relocation_arrival":
            inventory_before = inventory[event.station_id]
            change_inventory(event.station_id, event.bike_count, event.timestamp)
            relocated_in[event.station_id] += event.bike_count
            relocation_bikes_in_transit -= event.bike_count
            record_trace(
                timestamp=event.timestamp,
                event_kind="relocation_in",
                station_id=event.station_id,
                counterparty_station_id=event.destination_id,
                bike_count=event.bike_count,
                inventory_before=inventory_before,
                inventory_after=inventory[event.station_id],
            )
        elif event.kind == "unconditional_return":
            inventory_before = inventory[event.station_id]
            change_inventory(event.station_id, 1, event.timestamp)
            unconditional_returns += 1
            record_trace(
                timestamp=event.timestamp,
                event_kind="unconditional_return",
                station_id=event.station_id,
                trip_id=event.trip_id,
                bike_count=1,
                inventory_before=inventory_before,
                inventory_after=inventory[event.station_id],
            )
        elif event.kind == "conditional_return":
            if event.trip_id in active_internal_trips:
                inventory_before = inventory[event.station_id]
                change_inventory(event.station_id, 1, event.timestamp)
                active_internal_trips.remove(event.trip_id)
                successful_internal_returns += 1
                record_trace(
                    timestamp=event.timestamp,
                    event_kind="conditional_return",
                    station_id=event.station_id,
                    trip_id=event.trip_id,
                    bike_count=1,
                    inventory_before=inventory_before,
                    inventory_after=inventory[event.station_id],
                )
        elif event.kind == "decision":
            decision_epochs += 1
            transfers = policy.plan(
                dict(inventory),
                max_bikes=config.max_bikes_per_decision,
            )
            moved_this_epoch = sum(transfer.bike_count for transfer in transfers)
            if moved_this_epoch > config.max_bikes_per_decision:
                raise SimulationError("정책이 판단 시점별 재배치 한도를 초과했습니다")
            max_relocation_actions_in_epoch = max(max_relocation_actions_in_epoch, len(transfers))
            if transfers:
                relocation_batches += 1
            for transfer in transfers:
                if transfer.bike_count <= 0:
                    raise SimulationError("재배치 대수는 양수여야 합니다")
                if transfer.distance_km < 0 or transfer.travel_minutes < 0:
                    raise SimulationError("재배치 거리와 시간은 음수일 수 없습니다")
                if transfer.from_station_id == transfer.to_station_id:
                    raise SimulationError("동일 대여소 간 재배치는 허용되지 않습니다")
                if (
                    transfer.from_station_id not in inventory
                    or transfer.to_station_id not in inventory
                ):
                    raise SimulationError("운영 대상 밖 대여소가 재배치에 포함됐습니다")
                donor_inventory_before = inventory[transfer.from_station_id]
                change_inventory(
                    transfer.from_station_id,
                    -transfer.bike_count,
                    event.timestamp,
                )
                relocated_out[transfer.from_station_id] += transfer.bike_count
                record_trace(
                    timestamp=event.timestamp,
                    event_kind="relocation_out",
                    station_id=transfer.from_station_id,
                    counterparty_station_id=transfer.to_station_id,
                    bike_count=transfer.bike_count,
                    inventory_before=donor_inventory_before,
                    inventory_after=inventory[transfer.from_station_id],
                    distance_km=transfer.distance_km,
                    travel_minutes=transfer.travel_minutes,
                )
                if transfer.travel_minutes == 0:
                    receiver_inventory_before = inventory[transfer.to_station_id]
                    change_inventory(
                        transfer.to_station_id,
                        transfer.bike_count,
                        event.timestamp,
                    )
                    relocated_in[transfer.to_station_id] += transfer.bike_count
                    record_trace(
                        timestamp=event.timestamp,
                        event_kind="relocation_in",
                        station_id=transfer.to_station_id,
                        counterparty_station_id=transfer.from_station_id,
                        bike_count=transfer.bike_count,
                        inventory_before=receiver_inventory_before,
                        inventory_after=inventory[transfer.to_station_id],
                        distance_km=transfer.distance_km,
                        travel_minutes=transfer.travel_minutes,
                    )
                else:
                    arrival_at = event.timestamp + timedelta(minutes=transfer.travel_minutes)
                    relocation_bikes_in_transit += transfer.bike_count
                    if arrival_at < config.end:
                        arrival_event = ReplayEvent(
                            timestamp=arrival_at,
                            priority=0,
                            sequence=next_sequence,
                            kind="relocation_arrival",
                            station_id=transfer.to_station_id,
                            destination_id=transfer.from_station_id,
                            bike_count=transfer.bike_count,
                        )
                        heapq.heappush(
                            event_queue,
                            (
                                arrival_event.timestamp,
                                arrival_event.priority,
                                arrival_event.sequence,
                                arrival_event,
                            ),
                        )
                        next_sequence += 1
                relocation_actions += 1
                bikes_moved += transfer.bike_count
                relocation_distance_km += transfer.distance_km
                relocation_vehicle_minutes += transfer.travel_minutes
            max_bikes_moved_in_epoch = max(max_bikes_moved_in_epoch, moved_this_epoch)
        elif event.kind == "rental":
            observed_requests += 1
            requests[event.station_id] += 1
            inventory_before = inventory[event.station_id]
            if inventory[event.station_id] == 0:
                failed_rentals += 1
                record_trace(
                    timestamp=event.timestamp,
                    event_kind="rental",
                    station_id=event.station_id,
                    counterparty_station_id=event.destination_id,
                    trip_id=event.trip_id,
                    bike_count=1,
                    inventory_before=inventory_before,
                    inventory_after=inventory_before,
                    rental_outcome="failed",
                )
                continue
            change_inventory(event.station_id, -1, event.timestamp)
            successful_rentals += 1
            successes[event.station_id] += 1
            record_trace(
                timestamp=event.timestamp,
                event_kind="rental",
                station_id=event.station_id,
                counterparty_station_id=event.destination_id,
                trip_id=event.trip_id,
                bike_count=1,
                inventory_before=inventory_before,
                inventory_after=inventory[event.station_id],
                rental_outcome="successful",
            )
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
        - relocation_bikes_in_transit
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
        max_relocation_actions_in_epoch=max_relocation_actions_in_epoch,
        relocation_distance_km=round(relocation_distance_km, 3),
        relocation_vehicle_minutes=round(relocation_vehicle_minutes, 3),
        relocation_bikes_in_transit_at_end=relocation_bikes_in_transit,
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
    event_trace = None
    if collect_trace:
        event_trace = pl.DataFrame(
            trace_records,
            schema_overrides={
                "timestamp": pl.Datetime,
                "trace_sequence": pl.Int64,
                "trip_id": pl.Int64,
                "bike_count": pl.Int64,
                "inventory_before": pl.Int64,
                "inventory_after": pl.Int64,
                "distance_km": pl.Float64,
                "travel_minutes": pl.Float64,
            },
        ).sort("timestamp", "trace_sequence")
    return SimulationRun(
        metrics=metrics,
        station_metrics=station_metrics,
        event_trace=event_trace,
    )


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


def build_spatial_policy_comparison(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    evaluation_start: datetime,
    evaluation_end: datetime,
    decision_interval_minutes: int,
    max_bikes_per_decision: int,
    comparison_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> SpatialSimulationExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"공간 시뮬레이션 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise SimulationError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise SimulationError("좌표 대여소 ID가 중복됐습니다")

    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    eligible_station_ids = set(coordinates)
    all_actionable_ids = set(
        station_hour.filter(pl.col("actionable"))["station_id"].unique().to_list()
    )
    excluded_station_ids = tuple(
        sorted(
            all_actionable_ids - eligible_station_ids,
            key=lambda value: int(value),
        )
    )
    config = SimulationConfig(
        start=evaluation_start,
        end=evaluation_end,
        decision_interval_minutes=decision_interval_minutes,
        max_bikes_per_decision=max_bikes_per_decision,
    )
    scenario = build_replay_scenario(
        trips,
        station_hour,
        config,
        eligible_station_ids=eligible_station_ids,
    )
    policies: tuple[tuple[str, RelocationPolicy], ...] = (
        ("P0 재배치 없음", NoRelocationPolicy()),
        (
            "P1 즉시 임계값",
            StaticThresholdPolicy(
                lower_threshold=2,
                target_bikes=5,
                upper_threshold=8,
            ),
        ),
        (
            "P2 거리·시간 반영",
            GreedyNearestPolicy(
                coordinates=coordinates,
                lower_threshold=2,
                target_bikes=5,
                upper_threshold=8,
                max_actions_per_decision=2,
                vehicle_capacity=20,
                average_speed_kmh=15.0,
                road_distance_factor=1.3,
                handling_minutes_per_bike=0.75,
            ),
        ),
    )
    runs = [(label, simulate_replay(scenario, policy)) for label, policy in policies]
    evaluation_records = tuple(
        {"policy_label": label, **asdict(run.metrics)} for label, run in runs
    )
    comparison_frame = pl.DataFrame(evaluation_records)
    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame.write_csv(comparison_csv_path)
    _plot_spatial_comparison(comparison_frame, figure_path)

    baseline = runs[0][1].metrics
    comparisons = {}
    for label, run in runs[1:]:
        metrics = run.metrics
        failures_avoided = baseline.failed_rentals - metrics.failed_rentals
        empty_hours_reduced = baseline.empty_station_hours - metrics.empty_station_hours
        comparisons[metrics.policy_name] = {
            "failures_avoided_vs_p0": float(failures_avoided),
            "failure_reduction_rate_vs_p0": _rate(failures_avoided, baseline.failed_rentals),
            "service_rate_percentage_point_vs_p0": round(
                (metrics.service_rate - baseline.service_rate) * 100,
                3,
            ),
            "empty_hours_reduced_vs_p0": round(empty_hours_reduced, 3),
            "empty_time_reduction_rate_vs_p0": _rate(
                empty_hours_reduced, baseline.empty_station_hours
            ),
            "policy_label": label,
        }
    p1 = runs[1][1].metrics
    p2 = runs[2][1].metrics
    comparisons["greedy_nearest"].update(
        {
            "additional_failures_vs_p1": float(p2.failed_rentals - p1.failed_rentals),
            "additional_empty_hours_vs_p1": round(
                p2.empty_station_hours - p1.empty_station_hours,
                3,
            ),
        }
    )
    assumptions = {
        "lower_threshold": 2,
        "target_bikes": 5,
        "upper_threshold": 8,
        "decision_interval_minutes": decision_interval_minutes,
        "max_bikes_per_decision": max_bikes_per_decision,
        "max_direct_trips_per_decision": 2,
        "vehicle_capacity": 20,
        "average_speed_kmh": 15.0,
        "road_distance_factor": 1.3,
        "handling_minutes_per_bike": 0.75,
    }
    output_files = {
        "policy_comparison": str(comparison_csv_path),
        "comparison_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = SpatialSimulationExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="observed_trip_replay_with_delayed_direct_relocation",
        evaluation_window=(f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}"),
        stations=len(eligible_station_ids),
        coordinate_file=str(coordinate_path),
        excluded_station_ids=excluded_station_ids,
        assumptions=assumptions,
        evaluation_runs=evaluation_records,
        comparisons=comparisons,
        limitations=(
            "공개 대여이력에 기록된 성공 요청만 재생하고 미관측 잠재 수요는 제외한다.",
            "P2 차량은 판단 시점에 공급 대여소에서 바로 출발해 첫 접근 이동은 제외한다.",
            "도로 보정 직선거리와 고정 평균속도를 사용하며 실제 교통·신호는 반영하지 않는다.",
            "명목 거치대 수를 하드 용량으로 쓰지 않아 반납 실패는 계산하지 않는다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_spatial_markdown(experiment), encoding="utf-8")
    return experiment


def run_spatial_sensitivity(
    scenario: ReplayScenario,
    coordinates: dict[str, tuple[float, float]],
    *,
    action_counts: tuple[int, ...] = (1, 2, 3),
    speeds_kmh: tuple[float, ...] = (10.0, 15.0, 20.0),
    vehicle_capacities: tuple[int, ...] = (10, 20),
) -> pl.DataFrame:
    if not action_counts or not speeds_kmh or not vehicle_capacities:
        raise ValueError("민감도 요인에는 각각 하나 이상의 값이 필요합니다")
    if any(value <= 0 for value in (*action_counts, *speeds_kmh, *vehicle_capacities)):
        raise ValueError("민감도 요인 값은 모두 양수여야 합니다")
    required_budget = max(action_counts) * max(vehicle_capacities)
    if scenario.config.max_bikes_per_decision < required_budget:
        raise SimulationError(
            "시나리오 이동 한도가 민감도 최대 조합보다 작습니다: "
            f"{scenario.config.max_bikes_per_decision} < {required_budget}"
        )

    baseline = simulate_replay(scenario, NoRelocationPolicy()).metrics
    records: list[dict[str, Any]] = []
    for actions in action_counts:
        for capacity in vehicle_capacities:
            for speed in speeds_kmh:
                policy = GreedyNearestPolicy(
                    coordinates=coordinates,
                    lower_threshold=2,
                    target_bikes=5,
                    upper_threshold=8,
                    max_actions_per_decision=actions,
                    vehicle_capacity=capacity,
                    average_speed_kmh=speed,
                    road_distance_factor=1.3,
                    handling_minutes_per_bike=0.75,
                    name=f"greedy_a{actions}_c{capacity}_s{speed:g}",
                )
                metrics = simulate_replay(scenario, policy).metrics
                failures_avoided = baseline.failed_rentals - metrics.failed_rentals
                records.append(
                    {
                        "scenario_id": policy.name,
                        "max_actions_per_decision": actions,
                        "average_speed_kmh": float(speed),
                        "vehicle_capacity": capacity,
                        **asdict(metrics),
                        "failures_avoided_vs_p0": failures_avoided,
                        "failure_reduction_rate_vs_p0": _rate(
                            failures_avoided, baseline.failed_rentals
                        ),
                        "service_rate_percentage_point_vs_p0": round(
                            (metrics.service_rate - baseline.service_rate) * 100,
                            3,
                        ),
                        "empty_hours_reduced_vs_p0": round(
                            baseline.empty_station_hours - metrics.empty_station_hours,
                            3,
                        ),
                        "failures_avoided_per_100km": round(
                            100 * _rate(failures_avoided, metrics.relocation_distance_km),
                            3,
                        ),
                    }
                )
    return pl.DataFrame(records).sort(
        "max_actions_per_decision", "vehicle_capacity", "average_speed_kmh"
    )


def build_spatial_sensitivity(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    evaluation_start: datetime,
    evaluation_end: datetime,
    decision_interval_minutes: int,
    action_counts: tuple[int, ...],
    speeds_kmh: tuple[float, ...],
    vehicle_capacities: tuple[int, ...],
    comparison_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> SensitivityExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"민감도 분석 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise SimulationError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise SimulationError("좌표 대여소 ID가 중복됐습니다")

    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    eligible_station_ids = set(coordinates)
    all_actionable_ids = set(
        station_hour.filter(pl.col("actionable"))["station_id"].unique().to_list()
    )
    excluded_station_ids = tuple(
        sorted(all_actionable_ids - eligible_station_ids, key=lambda value: int(value))
    )
    max_bikes_per_decision = max(action_counts) * max(vehicle_capacities)
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(
            start=evaluation_start,
            end=evaluation_end,
            decision_interval_minutes=decision_interval_minutes,
            max_bikes_per_decision=max_bikes_per_decision,
        ),
        eligible_station_ids=eligible_station_ids,
    )
    baseline_metrics = simulate_replay(scenario, NoRelocationPolicy()).metrics
    frame = run_spatial_sensitivity(
        scenario,
        coordinates,
        action_counts=action_counts,
        speeds_kmh=speeds_kmh,
        vehicle_capacities=vehicle_capacities,
    )
    service_best = frame.sort(
        "failed_rentals", "relocation_vehicle_minutes", "relocation_distance_km"
    ).row(0, named=True)
    service_best["equivalent_vehicle_capacities"] = sorted(
        frame.filter(pl.col("failed_rentals") == service_best["failed_rentals"])["vehicle_capacity"]
        .unique()
        .to_list()
    )
    empty_time_best = frame.sort(
        "empty_station_hours", "failed_rentals", "relocation_vehicle_minutes"
    ).row(0, named=True)
    empty_time_best["equivalent_vehicle_capacities"] = sorted(
        frame.filter(pl.col("empty_station_hours") == empty_time_best["empty_station_hours"])[
            "vehicle_capacity"
        ]
        .unique()
        .to_list()
    )
    distance_efficiency_best = frame.sort(
        "failures_avoided_per_100km",
        "failed_rentals",
        descending=[True, False],
    ).row(0, named=True)
    default_rows = frame.filter(
        (pl.col("max_actions_per_decision") == 2)
        & (pl.col("average_speed_kmh") == 15.0)
        & (pl.col("vehicle_capacity") == 20)
    )
    default_scenario = (
        default_rows.row(0, named=True) if default_rows.height == 1 else frame.row(0, named=True)
    )
    capacity_failure_spread = (
        frame.group_by("max_actions_per_decision", "average_speed_kmh")
        .agg((pl.col("failed_rentals").max() - pl.col("failed_rentals").min()).alias("spread"))[
            "spread"
        ]
        .max()
    )
    speed_failure_spread = (
        frame.group_by("max_actions_per_decision", "vehicle_capacity")
        .agg((pl.col("failed_rentals").max() - pl.col("failed_rentals").min()).alias("spread"))[
            "spread"
        ]
        .max()
    )
    action_failure_spread = (
        frame.group_by("average_speed_kmh", "vehicle_capacity")
        .agg((pl.col("failed_rentals").max() - pl.col("failed_rentals").min()).alias("spread"))[
            "spread"
        ]
        .max()
    )
    factor_findings = {
        "max_failed_rental_spread_from_capacity": int(capacity_failure_spread or 0),
        "max_failed_rental_spread_from_speed": int(speed_failure_spread or 0),
        "max_failed_rental_spread_from_actions": int(action_failure_spread or 0),
        "capacity_non_binding_reason": (
            "수신 대여소 목표가 5대라 직접 운송 1회의 필요량이 최대 5대이며, "
            "비교한 적재량 10대와 20대는 모두 이를 초과한다."
        ),
    }
    marginal_actions = _calculate_action_marginals(frame)

    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(comparison_csv_path)
    _plot_spatial_sensitivity(frame, figure_path)
    output_files = {
        "sensitivity_grid": str(comparison_csv_path),
        "sensitivity_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = SensitivityExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="observed_trip_replay_spatial_factorial_sensitivity",
        evaluation_window=f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}",
        stations=len(eligible_station_ids),
        coordinate_file=str(coordinate_path),
        excluded_station_ids=excluded_station_ids,
        factors={
            "max_direct_trips_per_decision": list(action_counts),
            "average_speed_kmh": list(speeds_kmh),
            "vehicle_capacity": list(vehicle_capacities),
            "decision_interval_minutes": decision_interval_minutes,
            "scenario_max_bikes_per_decision": max_bikes_per_decision,
            "lower_target_upper_thresholds": [2, 5, 8],
            "road_distance_factor": 1.3,
            "handling_minutes_per_bike": 0.75,
        },
        baseline=asdict(baseline_metrics),
        runs=tuple(frame.to_dicts()),
        service_best=service_best,
        empty_time_best=empty_time_best,
        distance_efficiency_best=distance_efficiency_best,
        default_scenario=default_scenario,
        factor_findings=factor_findings,
        marginal_actions=marginal_actions,
        limitations=(
            "공개 대여이력에 기록된 성공 요청만 재생하고 미관측 잠재 수요는 제외한다.",
            "직접 운송 횟수는 운영능력의 대리변수이며 실제 독립 차량 대수와 같지 않다.",
            "차량은 공급 대여소에서 바로 출발해 첫 접근 이동과 차량 간 경로 연결을 제외한다.",
            "거리 효율은 비용이 아니라 100km당 방지 실패이며 인건비·차량비를 포함하지 않는다.",
            "고정 평균속도와 직선거리 보정값을 사용하며 실제 교통·신호는 반영하지 않는다.",
            "명목 거치대 수를 하드 용량으로 쓰지 않아 반납 실패는 계산하지 않는다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_sensitivity_markdown(experiment), encoding="utf-8")
    return experiment


def _calculate_action_marginals(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for speed in sorted(frame["average_speed_kmh"].unique().to_list()):
        for capacity in sorted(frame["vehicle_capacity"].unique().to_list()):
            subset = frame.filter(
                (pl.col("average_speed_kmh") == speed) & (pl.col("vehicle_capacity") == capacity)
            ).sort("max_actions_per_decision")
            rows = subset.to_dicts()
            for previous, current in zip(rows, rows[1:], strict=False):
                additional_distance = round(
                    current["relocation_distance_km"] - previous["relocation_distance_km"],
                    3,
                )
                additional_failures_avoided = previous["failed_rentals"] - current["failed_rentals"]
                records.append(
                    {
                        "average_speed_kmh": speed,
                        "vehicle_capacity": capacity,
                        "from_actions": previous["max_actions_per_decision"],
                        "to_actions": current["max_actions_per_decision"],
                        "additional_failures_avoided": additional_failures_avoided,
                        "additional_distance_km": additional_distance,
                        "additional_vehicle_minutes": round(
                            current["relocation_vehicle_minutes"]
                            - previous["relocation_vehicle_minutes"],
                            3,
                        ),
                        "additional_failures_avoided_per_100km": round(
                            100 * _rate(additional_failures_avoided, additional_distance),
                            3,
                        ),
                    }
                )
    return tuple(records)


def run_daily_policy_comparison(
    trips: pl.DataFrame,
    station_hour: pl.DataFrame,
    coordinates: dict[str, tuple[float, float]],
    *,
    analysis_start: datetime,
    analysis_end: datetime,
) -> tuple[pl.DataFrame, tuple[dict[str, Any], ...]]:
    if analysis_end <= analysis_start:
        raise ValueError("일별 분석 종료 시각은 시작 시각보다 늦어야 합니다")
    if any(
        value != 0
        for value in (
            analysis_start.hour,
            analysis_start.minute,
            analysis_start.second,
            analysis_end.hour,
            analysis_end.minute,
            analysis_end.second,
        )
    ):
        raise ValueError("일별 분석 시작과 종료는 자정이어야 합니다")
    station_ids = set(coordinates)
    policies: tuple[tuple[int, str, RelocationPolicy], ...] = (
        (0, "P0 재배치 없음", NoRelocationPolicy()),
        (
            1,
            "P2 기존 2회·15km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=2,
                vehicle_capacity=20,
                average_speed_kmh=15.0,
                name="greedy_default",
            ),
        ),
        (
            2,
            "P2 서비스 3회·20km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=3,
                vehicle_capacity=10,
                average_speed_kmh=20.0,
                name="greedy_service",
            ),
        ),
    )
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    day_start = analysis_start
    while day_start < analysis_end:
        day_end = day_start + timedelta(days=1)
        initial_rows = station_hour.filter(
            (pl.col("timestamp") == day_start)
            & pl.col("actionable")
            & pl.col("inventory_observed")
            & pl.col("station_id").is_in(station_ids)
        )
        available_ids = set(initial_rows["station_id"].unique().to_list())
        if available_ids != station_ids:
            missing_ids = tuple(sorted(station_ids - available_ids))
            excluded.append(
                {
                    "date": day_start.date().isoformat(),
                    "reason": "incomplete_midnight_inventory",
                    "available_stations": len(available_ids),
                    "missing_station_ids": missing_ids,
                }
            )
            day_start = day_end
            continue
        scenario = build_replay_scenario(
            trips,
            station_hour,
            SimulationConfig(
                start=day_start,
                end=day_end,
                decision_interval_minutes=60,
                max_bikes_per_decision=40,
            ),
            eligible_station_ids=station_ids,
        )
        day_type = "주말" if day_start.weekday() >= 5 else "주중"
        for policy_order, label, policy in policies:
            metrics = simulate_replay(scenario, policy).metrics
            records.append(
                {
                    "date": day_start.date().isoformat(),
                    "weekday_number": day_start.weekday(),
                    "day_type": day_type,
                    "policy_order": policy_order,
                    "policy_label": label,
                    **asdict(metrics),
                }
            )
        day_start = day_end
    if not records:
        raise SimulationError("유효한 일별 시뮬레이션 날짜가 없습니다")

    baseline_by_date = {
        str(record["date"]): record
        for record in records
        if record["policy_name"] == "no_relocation"
    }
    default_by_date = {
        str(record["date"]): record
        for record in records
        if record["policy_name"] == "greedy_default"
    }
    for record in records:
        baseline = baseline_by_date[str(record["date"])]
        failures_avoided = baseline["failed_rentals"] - record["failed_rentals"]
        record["failures_avoided_vs_p0"] = failures_avoided
        record["service_rate_percentage_point_vs_p0"] = round(
            (record["service_rate"] - baseline["service_rate"]) * 100,
            3,
        )
        record["empty_hours_reduced_vs_p0"] = round(
            baseline["empty_station_hours"] - record["empty_station_hours"],
            3,
        )
        record["additional_failures_avoided_vs_default"] = (
            default_by_date[str(record["date"])]["failed_rentals"] - record["failed_rentals"]
            if record["policy_name"] == "greedy_service"
            else 0
        )
    frame = pl.DataFrame(records).sort("date", "policy_order")
    request_counts = frame.group_by("date").agg(
        pl.col("observed_requests").n_unique().alias("unique_request_counts")
    )
    if request_counts["unique_request_counts"].max() != 1:
        raise SimulationError("동일 날짜의 정책별 관측 요청 수가 다릅니다")
    return frame, tuple(excluded)


def build_temporal_robustness(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    analysis_start: datetime,
    analysis_end: datetime,
    daily_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> TemporalRobustnessExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"시간적 강건성 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise SimulationError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise SimulationError("좌표 대여소 ID가 중복됐습니다")
    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    daily_frame, excluded_days = run_daily_policy_comparison(
        trips,
        station_hour,
        coordinates,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )
    group_summaries = _summarize_temporal_groups(daily_frame)
    effect_consistency = _summarize_effect_consistency(daily_frame)
    daily_csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily_frame.write_csv(daily_csv_path)
    _plot_temporal_robustness(daily_frame, group_summaries, figure_path)
    output_files = {
        "daily_comparison": str(daily_csv_path),
        "robustness_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = TemporalRobustnessExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="independent_daily_observed_trip_replay",
        analysis_window=f"{analysis_start.isoformat()} <= t < {analysis_end.isoformat()}",
        stations=len(coordinates),
        coordinate_file=str(coordinate_path),
        valid_days=daily_frame["date"].n_unique(),
        excluded_days=excluded_days,
        policies={
            "no_relocation": {"label": "P0 재배치 없음"},
            "greedy_default": {
                "label": "P2 기존 2회·15km/h",
                "max_direct_trips_per_hour": 2,
                "average_speed_kmh": 15.0,
                "vehicle_capacity": 20,
            },
            "greedy_service": {
                "label": "P2 서비스 3회·20km/h",
                "max_direct_trips_per_hour": 3,
                "average_speed_kmh": 20.0,
                "vehicle_capacity": 10,
            },
        },
        daily_runs=tuple(daily_frame.to_dicts()),
        group_summaries=group_summaries,
        effect_consistency=effect_consistency,
        limitations=(
            "각 날짜를 00시 관측 재고에서 독립적으로 시작해 전날 정책 효과를 이어받지 않는다.",
            "월 집계 P0에는 자정 이전의 실제 운영 결과가 시작 재고에 포함되며, "
            "30일 연속 무재배치가 아니다.",
            "따라서 날짜별 P0 월 성공률은 기존 5일 연속 P0 성공률과 직접 비교할 수 없다.",
            "11월 3~21일은 임계값 선택에 사용돼 월 전체 결과는 순수 홀드아웃 추정치가 아니다.",
            "공개 대여이력에 기록된 성공 요청만 재생하고 미관측 잠재 수요는 제외한다.",
            "차량의 첫 접근 이동과 차량 간 연속 경로, 실제 교통·신호를 반영하지 않는다.",
            "주중·주말 차이는 날씨·행사·공휴일 등 교란요인을 통제한 인과효과가 아니다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_temporal_markdown(experiment), encoding="utf-8")
    return experiment


def _summarize_temporal_groups(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    policy_rows = (
        frame.select("policy_order", "policy_name", "policy_label").unique().sort("policy_order")
    )
    for policy in policy_rows.iter_rows(named=True):
        for day_type in ("주중", "주말"):
            subset = frame.filter(
                (pl.col("policy_name") == policy["policy_name"]) & (pl.col("day_type") == day_type)
            )
            if subset.is_empty():
                continue
            total_requests = int(subset["observed_requests"].sum())
            total_successes = int(subset["successful_rentals"].sum())
            improvements = subset["failures_avoided_vs_p0"]
            records.append(
                {
                    "policy_order": policy["policy_order"],
                    "policy_name": policy["policy_name"],
                    "policy_label": policy["policy_label"],
                    "day_type": day_type,
                    "days": subset.height,
                    "observed_requests": total_requests,
                    "successful_rentals": total_successes,
                    "failed_rentals": int(subset["failed_rentals"].sum()),
                    "weighted_service_rate": _rate(total_successes, total_requests),
                    "empty_station_hours": round(subset["empty_station_hours"].sum(), 3),
                    "bikes_moved": int(subset["bikes_moved"].sum()),
                    "relocation_distance_km": round(subset["relocation_distance_km"].sum(), 3),
                    "relocation_vehicle_minutes": round(
                        subset["relocation_vehicle_minutes"].sum(), 3
                    ),
                    "failures_avoided_vs_p0": int(improvements.sum()),
                    "median_daily_failures_avoided": float(improvements.median()),
                    "days_better_than_p0": int((improvements > 0).sum()),
                    "days_tied_with_p0": int((improvements == 0).sum()),
                    "days_worse_than_p0": int((improvements < 0).sum()),
                }
            )
    return tuple(records)


def _summarize_effect_consistency(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for policy_name in ("greedy_default", "greedy_service"):
        subset = frame.filter(pl.col("policy_name") == policy_name)
        improvements = subset["failures_avoided_vs_p0"]
        worst = subset.sort("failures_avoided_vs_p0", "date").row(0, named=True)
        best = subset.sort("failures_avoided_vs_p0", "date", descending=[True, False]).row(
            0, named=True
        )
        summaries[policy_name] = {
            "days": subset.height,
            "days_better_than_p0": int((improvements > 0).sum()),
            "days_tied_with_p0": int((improvements == 0).sum()),
            "days_worse_than_p0": int((improvements < 0).sum()),
            "minimum_daily_failures_avoided": int(improvements.min()),
            "median_daily_failures_avoided": float(improvements.median()),
            "maximum_daily_failures_avoided": int(improvements.max()),
            "weakest_date": worst["date"],
            "strongest_date": best["date"],
            "reversal_dates": tuple(
                subset.filter(pl.col("failures_avoided_vs_p0") < 0)["date"].to_list()
            ),
            "tie_dates": tuple(
                subset.filter(pl.col("failures_avoided_vs_p0") == 0)["date"].to_list()
            ),
        }
    service = frame.filter(pl.col("policy_name") == "greedy_service")
    incremental = service["additional_failures_avoided_vs_default"]
    weakest = service.sort("additional_failures_avoided_vs_default", "date").row(0, named=True)
    strongest = service.sort(
        "additional_failures_avoided_vs_default", "date", descending=[True, False]
    ).row(0, named=True)
    summaries["greedy_service_vs_default"] = {
        "days": service.height,
        "days_better_than_default": int((incremental > 0).sum()),
        "days_tied_with_default": int((incremental == 0).sum()),
        "days_worse_than_default": int((incremental < 0).sum()),
        "minimum_daily_additional_failures_avoided": int(incremental.min()),
        "median_daily_additional_failures_avoided": float(incremental.median()),
        "maximum_daily_additional_failures_avoided": int(incremental.max()),
        "weakest_date": weakest["date"],
        "strongest_date": strongest["date"],
        "worse_dates": tuple(
            service.filter(pl.col("additional_failures_avoided_vs_default") < 0)["date"].to_list()
        ),
        "tie_dates": tuple(
            service.filter(pl.col("additional_failures_avoided_vs_default") == 0)["date"].to_list()
        ),
    }
    return summaries


def run_station_equity_comparison(
    scenario: ReplayScenario,
    coordinates: dict[str, tuple[float, float]],
) -> tuple[pl.DataFrame, tuple[tuple[str, SimulationRun], ...]]:
    if set(scenario.initial_inventory) != set(coordinates):
        raise SimulationError("공간 형평성 시나리오와 좌표 대여소 범위가 다릅니다")
    policies: tuple[tuple[str, RelocationPolicy], ...] = (
        ("P0 재배치 없음", NoRelocationPolicy()),
        (
            "P2 기존 2회·15km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=2,
                vehicle_capacity=20,
                average_speed_kmh=15.0,
                name="greedy_default",
            ),
        ),
        (
            "P2 서비스 3회·20km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=3,
                vehicle_capacity=10,
                average_speed_kmh=20.0,
                name="greedy_service",
            ),
        ),
    )
    runs = tuple((label, simulate_replay(scenario, policy)) for label, policy in policies)
    baseline = runs[0][1].station_metrics.select(
        "station_id",
        "station_name",
        "requests",
        pl.col("successful_rentals").alias("p0_successful_rentals"),
        pl.col("failed_rentals").alias("p0_failed_rentals"),
        pl.col("service_rate").alias("p0_service_rate"),
        pl.col("empty_minutes").alias("p0_empty_minutes"),
        pl.col("empty_rate").alias("p0_empty_rate"),
        pl.col("initial_bikes").alias("initial_bikes"),
        pl.col("final_bikes").alias("p0_final_bikes"),
    )
    frame = baseline
    for prefix, run in (("default", runs[1][1]), ("service", runs[2][1])):
        policy_frame = run.station_metrics.select(
            "station_id",
            pl.col("requests").alias(f"{prefix}_requests"),
            pl.col("successful_rentals").alias(f"{prefix}_successful_rentals"),
            pl.col("failed_rentals").alias(f"{prefix}_failed_rentals"),
            pl.col("service_rate").alias(f"{prefix}_service_rate"),
            pl.col("empty_minutes").alias(f"{prefix}_empty_minutes"),
            pl.col("empty_rate").alias(f"{prefix}_empty_rate"),
            pl.col("relocated_in").alias(f"{prefix}_relocated_in"),
            pl.col("relocated_out").alias(f"{prefix}_relocated_out"),
            pl.col("final_bikes").alias(f"{prefix}_final_bikes"),
        )
        frame = frame.join(policy_frame, on="station_id", how="inner", validate="1:1")
        if not (frame["requests"] == frame[f"{prefix}_requests"]).all():
            raise SimulationError(f"{prefix} 정책의 대여소별 요청 수가 P0와 다릅니다")
        frame = frame.drop(f"{prefix}_requests").with_columns(
            (pl.col("p0_failed_rentals") - pl.col(f"{prefix}_failed_rentals")).alias(
                f"{prefix}_failures_avoided_vs_p0"
            ),
            ((pl.col(f"{prefix}_service_rate") - pl.col("p0_service_rate")) * 100)
            .round(3)
            .alias(f"{prefix}_service_rate_pp_vs_p0"),
            ((pl.col("p0_empty_minutes") - pl.col(f"{prefix}_empty_minutes")) / 60)
            .round(3)
            .alias(f"{prefix}_empty_hours_reduced_vs_p0"),
        )
    coordinate_frame = pl.DataFrame(
        [
            {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
            }
            for station_id, (latitude, longitude) in coordinates.items()
        ]
    )
    frame = frame.join(coordinate_frame, on="station_id", how="inner", validate="1:1")
    frame = frame.with_columns(
        (pl.col("default_failed_rentals") - pl.col("service_failed_rentals")).alias(
            "service_additional_failures_avoided_vs_default"
        )
    ).sort("station_id")
    if frame.height != len(coordinates):
        raise SimulationError("공간 형평성 대여소 조인 후 행 수가 달라졌습니다")
    return frame, runs


def build_station_equity(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    evaluation_start: datetime,
    evaluation_end: datetime,
    station_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> StationEquityExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"공간 형평성 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise SimulationError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise SimulationError("좌표 대여소 ID가 중복됐습니다")
    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(
            start=evaluation_start,
            end=evaluation_end,
            decision_interval_minutes=60,
            max_bikes_per_decision=40,
        ),
        eligible_station_ids=set(coordinates),
    )
    station_frame, runs = run_station_equity_comparison(scenario, coordinates)
    equity_summaries = _summarize_station_equity(station_frame, runs)
    station_csv_path.parent.mkdir(parents=True, exist_ok=True)
    station_frame.write_csv(station_csv_path)
    _plot_station_equity(station_frame, runs, figure_path)
    output_files = {
        "station_comparison": str(station_csv_path),
        "equity_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = StationEquityExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="continuous_holdout_station_equity_replay",
        evaluation_window=f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}",
        stations=station_frame.height,
        coordinate_file=str(coordinate_path),
        policies={
            "no_relocation": {"label": runs[0][0]},
            "greedy_default": {
                "label": runs[1][0],
                "max_direct_trips_per_hour": 2,
                "average_speed_kmh": 15.0,
                "vehicle_capacity": 20,
            },
            "greedy_service": {
                "label": runs[2][0],
                "max_direct_trips_per_hour": 3,
                "average_speed_kmh": 20.0,
                "vehicle_capacity": 10,
            },
        },
        policy_runs=tuple({"policy_label": label, **asdict(run.metrics)} for label, run in runs),
        equity_summaries=equity_summaries,
        station_results=tuple(station_frame.to_dicts()),
        limitations=(
            "공개 대여이력의 성공 요청만 재생해 미관측 잠재 수요의 공간 분포는 알 수 없다.",
            "서비스 P2 조합은 같은 홀드아웃 민감도에서 선택돼 독립 검증 성과가 아니다.",
            "대여소별 실패 감소는 형평성의 운영 대리변수이며 인구·소득·교통약자를 포함하지 않는다.",
            "현재 좌표 스냅샷이 2025년 운영 당시 위치와 다를 수 있다.",
            "차량 첫 접근 이동·연속 경로·실제 교통과 반납 실패를 반영하지 않는다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_station_equity_markdown(experiment), encoding="utf-8")
    return experiment


def _summarize_station_equity(
    frame: pl.DataFrame,
    runs: tuple[tuple[str, SimulationRun], ...],
) -> dict[str, dict[str, Any]]:
    run_by_name = {run.metrics.policy_name: run for _, run in runs}
    active = frame.filter(pl.col("requests") > 0)
    summaries: dict[str, dict[str, Any]] = {}
    for prefix, policy_name in (
        ("default", "greedy_default"),
        ("service", "greedy_service"),
    ):
        avoided_column = f"{prefix}_failures_avoided_vs_p0"
        improvements = active[avoided_column]
        positive = frame.filter(pl.col(avoided_column) > 0).sort(avoided_column, descending=True)
        worsened = frame.filter(pl.col(avoided_column) < 0).sort(avoided_column)
        top_ten = positive.head(10)
        positive_total = int(positive[avoided_column].sum())
        top_ten_total = int(top_ten[avoided_column].sum())
        run = run_by_name[policy_name]
        summaries[policy_name] = {
            "stations": frame.height,
            "stations_with_requests": active.height,
            "stations_improved": int((improvements > 0).sum()),
            "stations_tied": int((improvements == 0).sum()),
            "stations_worsened": int((improvements < 0).sum()),
            "net_failures_avoided": int(frame[avoided_column].sum()),
            "positive_failures_avoided": positive_total,
            "failures_added_at_worsened_stations": int(-worsened[avoided_column].sum()),
            "top_10_positive_failures_avoided": top_ten_total,
            "top_10_positive_concentration": _rate(top_ten_total, positive_total),
            "overall_service_rate": run.metrics.service_rate,
            "p10_station_service_rate": run.metrics.p10_station_service_rate,
            "p10_service_rate_pp_vs_p0": round(
                (
                    run.metrics.p10_station_service_rate
                    - run_by_name["no_relocation"].metrics.p10_station_service_rate
                )
                * 100,
                3,
            ),
            "worst_station_id": run.metrics.worst_station_id,
            "worst_station_name": run.metrics.worst_station_name,
            "worst_station_service_rate": run.metrics.worst_station_service_rate,
            "relocation_out_and_worsened_stations": frame.filter(
                (pl.col(f"{prefix}_relocated_out") > 0) & (pl.col(avoided_column) < 0)
            ).height,
            "top_10_beneficiary_stations": tuple(
                {
                    "station_id": row["station_id"],
                    "station_name": row["station_name"],
                    "requests": row["requests"],
                    "failures_avoided": row[avoided_column],
                    "relocated_in": row[f"{prefix}_relocated_in"],
                    "relocated_out": row[f"{prefix}_relocated_out"],
                }
                for row in top_ten.iter_rows(named=True)
            ),
            "worsened_stations": tuple(
                {
                    "station_id": row["station_id"],
                    "station_name": row["station_name"],
                    "requests": row["requests"],
                    "failures_added": -row[avoided_column],
                    "relocated_in": row[f"{prefix}_relocated_in"],
                    "relocated_out": row[f"{prefix}_relocated_out"],
                }
                for row in worsened.iter_rows(named=True)
            ),
        }
    return summaries


def run_request_transition_trace(
    scenario: ReplayScenario,
    coordinates: dict[str, tuple[float, float]],
) -> tuple[
    pl.DataFrame,
    dict[str, dict[str, Any]],
    tuple[tuple[str, SimulationRun], ...],
]:
    if set(scenario.initial_inventory) != set(coordinates):
        raise SimulationError("요청 전환 추적 시나리오와 좌표 대여소 범위가 다릅니다")
    policies: tuple[tuple[str, RelocationPolicy], ...] = (
        ("P0 재배치 없음", NoRelocationPolicy()),
        (
            "P2 기존 2회·15km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=2,
                vehicle_capacity=20,
                average_speed_kmh=15.0,
                name="greedy_default",
            ),
        ),
        (
            "P2 서비스 3회·20km/h",
            GreedyNearestPolicy(
                coordinates=coordinates,
                max_actions_per_decision=3,
                vehicle_capacity=10,
                average_speed_kmh=20.0,
                name="greedy_service",
            ),
        ),
    )
    runs = tuple(
        (label, simulate_replay(scenario, policy, collect_trace=True)) for label, policy in policies
    )
    if any(run.event_trace is None for _, run in runs):
        raise SimulationError("요청 전환 추적 이벤트가 생성되지 않았습니다")
    baseline_run = runs[0][1]
    baseline_rentals = baseline_run.event_trace.filter(  # type: ignore[union-attr]
        pl.col("event_kind") == "rental"
    ).select(
        "trip_id",
        pl.col("timestamp").alias("request_at"),
        "station_id",
        pl.col("rental_outcome").alias("p0_outcome"),
        pl.col("inventory_before").alias("p0_inventory_before"),
    )
    baseline_station_failures = {
        str(row["station_id"]): int(row["failed_rentals"])
        for row in baseline_run.station_metrics.iter_rows(named=True)
    }
    harm_records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for label, run in runs[1:]:
        trace = run.event_trace
        if trace is None:
            raise SimulationError(f"{run.metrics.policy_name} trace가 없습니다")
        policy_rentals = trace.filter(pl.col("event_kind") == "rental").select(
            "trip_id",
            pl.col("timestamp").alias("policy_request_at"),
            pl.col("station_id").alias("policy_station_id"),
            pl.col("rental_outcome").alias("policy_outcome"),
            pl.col("inventory_before").alias("policy_inventory_before"),
        )
        transitions = baseline_rentals.join(
            policy_rentals,
            on="trip_id",
            how="inner",
            validate="1:1",
        )
        if transitions.height != baseline_rentals.height:
            raise SimulationError(f"{run.metrics.policy_name} 요청 trace 행 수가 P0와 다릅니다")
        if not (
            (transitions["request_at"] == transitions["policy_request_at"]).all()
            and (transitions["station_id"] == transitions["policy_station_id"]).all()
        ):
            raise SimulationError(f"{run.metrics.policy_name} 요청 trace 키가 P0와 다릅니다")
        rescued = transitions.filter(
            (pl.col("p0_outcome") == "failed") & (pl.col("policy_outcome") == "successful")
        )
        harmed = transitions.filter(
            (pl.col("p0_outcome") == "successful") & (pl.col("policy_outcome") == "failed")
        )
        net_failures_avoided = baseline_run.metrics.failed_rentals - run.metrics.failed_rentals
        if rescued.height - harmed.height != net_failures_avoided:
            raise SimulationError(f"{run.metrics.policy_name} rescue-harm 재조정이 맞지 않습니다")

        relocation_out = trace.filter(pl.col("event_kind") == "relocation_out").sort(
            "station_id", "timestamp", "trace_sequence"
        )
        out_by_station: dict[str, list[dict[str, Any]]] = {}
        for row in relocation_out.iter_rows(named=True):
            out_by_station.setdefault(str(row["station_id"]), []).append(row)
        policy_station_failures = {
            str(row["station_id"]): int(row["failed_rentals"])
            for row in run.station_metrics.iter_rows(named=True)
        }
        policy_harm_records: list[dict[str, Any]] = []
        for row in harmed.iter_rows(named=True):
            station_id = str(row["station_id"])
            request_at = row["request_at"]
            station_outs = out_by_station.get(station_id, [])
            out_times = [item["timestamp"] for item in station_outs]
            prior_index = bisect_left(out_times, request_at) - 1
            prior = station_outs[prior_index] if prior_index >= 0 else None
            minutes_since_out = (
                round((request_at - prior["timestamp"]).total_seconds() / 60, 3)
                if prior is not None
                else None
            )
            net_station_avoided = (
                baseline_station_failures[station_id] - policy_station_failures[station_id]
            )
            record = {
                "policy_name": run.metrics.policy_name,
                "policy_label": label,
                "trip_id": int(row["trip_id"]),
                "request_at": request_at.isoformat(),
                "station_id": station_id,
                "station_name": scenario.station_names[station_id],
                "p0_inventory_before": int(row["p0_inventory_before"]),
                "policy_inventory_before": int(row["policy_inventory_before"]),
                "station_net_failures_avoided_vs_p0": net_station_avoided,
                "station_net_status": (
                    "improved"
                    if net_station_avoided > 0
                    else "worsened"
                    if net_station_avoided < 0
                    else "tied"
                ),
                "has_prior_relocation_out": prior is not None,
                "prior_relocation_out_at": (
                    prior["timestamp"].isoformat() if prior is not None else None
                ),
                "minutes_since_prior_out": minutes_since_out,
                "within_60_minutes_of_prior_out": (
                    minutes_since_out is not None and minutes_since_out <= 60
                ),
                "prior_out_bikes": int(prior["bike_count"]) if prior is not None else None,
                "prior_out_inventory_before": (
                    int(prior["inventory_before"]) if prior is not None else None
                ),
                "prior_out_inventory_after": (
                    int(prior["inventory_after"]) if prior is not None else None
                ),
                "prior_out_destination_id": (
                    str(prior["counterparty_station_id"]) if prior is not None else None
                ),
            }
            policy_harm_records.append(record)
            harm_records.append(record)
        linked_records = [
            record for record in policy_harm_records if record["has_prior_relocation_out"]
        ]
        within_60 = [
            record for record in policy_harm_records if record["within_60_minutes_of_prior_out"]
        ]
        linked_minutes = sorted(
            float(record["minutes_since_prior_out"]) for record in linked_records
        )
        station_counts: dict[tuple[str, str], int] = {}
        for record in policy_harm_records:
            key = (str(record["station_id"]), str(record["station_name"]))
            station_counts[key] = station_counts.get(key, 0) + 1
        top_harm_stations = tuple(
            {
                "station_id": station_id,
                "station_name": station_name,
                "harm_requests": count,
                "net_failures_avoided_vs_p0": (
                    baseline_station_failures[station_id] - policy_station_failures[station_id]
                ),
            }
            for (station_id, station_name), count in sorted(
                station_counts.items(),
                key=lambda item: (-item[1], item[0][0]),
            )[:10]
        )
        summaries[run.metrics.policy_name] = {
            "observed_requests": transitions.height,
            "rescued_requests": rescued.height,
            "harmed_requests": harmed.height,
            "net_failures_avoided": net_failures_avoided,
            "reconciliation_residual": rescued.height - harmed.height - net_failures_avoided,
            "harm_with_prior_relocation_out": len(linked_records),
            "harm_without_prior_relocation_out": harmed.height - len(linked_records),
            "prior_out_link_rate": _rate(len(linked_records), harmed.height),
            "harm_within_60_minutes_of_prior_out": len(within_60),
            "within_60_minutes_rate_of_harm": _rate(len(within_60), harmed.height),
            "median_minutes_since_prior_out": (
                round(median(linked_minutes), 3) if linked_minutes else None
            ),
            "harm_at_net_worsened_stations": sum(
                record["station_net_status"] == "worsened" for record in policy_harm_records
            ),
            "harm_at_net_improved_stations": sum(
                record["station_net_status"] == "improved" for record in policy_harm_records
            ),
            "top_harm_stations": top_harm_stations,
        }
    harm_frame = pl.DataFrame(harm_records).sort("policy_name", "request_at", "trip_id")
    return harm_frame, summaries, runs


def build_harm_trace(
    *,
    trips_path: Path,
    station_hour_path: Path,
    coordinate_path: Path,
    evaluation_start: datetime,
    evaluation_end: datetime,
    harm_csv_path: Path,
    figure_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> HarmTraceExperiment:
    try:
        trips = pl.read_parquet(trips_path)
        station_hour = pl.read_parquet(station_hour_path)
        coordinate_frame = pl.read_csv(
            coordinate_path,
            schema_overrides={"station_id": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SimulationError(f"요청 전환 추적 입력을 읽지 못했습니다: {exc}") from exc
    required_coordinate_columns = {"station_id", "latitude", "longitude"}
    missing_columns = required_coordinate_columns - set(coordinate_frame.columns)
    if missing_columns:
        raise SimulationError(f"좌표 필수 열 누락: {sorted(missing_columns)}")
    if coordinate_frame["station_id"].n_unique() != coordinate_frame.height:
        raise SimulationError("좌표 대여소 ID가 중복됐습니다")
    coordinates = {
        str(row["station_id"]): (float(row["latitude"]), float(row["longitude"]))
        for row in coordinate_frame.iter_rows(named=True)
    }
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(
            start=evaluation_start,
            end=evaluation_end,
            decision_interval_minutes=60,
            max_bikes_per_decision=40,
        ),
        eligible_station_ids=set(coordinates),
    )
    harm_frame, summaries, runs = run_request_transition_trace(scenario, coordinates)
    harm_csv_path.parent.mkdir(parents=True, exist_ok=True)
    harm_frame.write_csv(harm_csv_path)
    _plot_harm_trace(harm_frame, summaries, figure_path)
    output_files = {
        "harm_requests": str(harm_csv_path),
        "harm_trace_figure": str(figure_path),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    experiment = HarmTraceExperiment(
        generated_at_utc=datetime.now(UTC).isoformat(),
        method="trip_id_counterfactual_transition_with_prior_outflow_trace",
        evaluation_window=f"{evaluation_start.isoformat()} <= t < {evaluation_end.isoformat()}",
        stations=len(coordinates),
        policies={
            "greedy_default": {"label": runs[1][0]},
            "greedy_service": {"label": runs[2][0]},
        },
        policy_runs=tuple({"policy_label": label, **asdict(run.metrics)} for label, run in runs),
        transition_summaries=summaries,
        harm_requests=tuple(harm_frame.to_dicts()),
        limitations=(
            "선행 재배치 유출 연결은 시간적 연관이며 해당 요청 실패의 인과 증명이 아니다.",
            "가장 최근 유출만 연결해 여러 차례 누적 유출과 네트워크 연쇄효과를 단순화한다.",
            "60분 기준은 정책 판단 주기와 같다는 운영상 기준이며 최적 임계시간이 아니다.",
            "공개 이력의 성공 요청만 재생해 미관측 잠재 수요 전환은 분석할 수 없다.",
            "서비스 P2는 같은 홀드아웃에서 선택돼 독립 검증 결과가 아니다.",
        ),
        output_files=output_files,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_harm_trace_markdown(experiment), encoding="utf-8")
    return experiment


def _render_harm_trace_markdown(experiment: HarmTraceExperiment) -> str:
    default = experiment.transition_summaries["greedy_default"]
    service = experiment.transition_summaries["greedy_service"]
    summary_rows = "\n".join(
        "| {label} | {rescued:,} | {harmed:,} | {net:+,} | {linked:,} "
        "({link_rate:.2%}) | {within:,} ({within_rate:.2%}) | {unlinked:,} | "
        "{median_minutes:,.1f} |".format(
            label=experiment.policies[policy_name]["label"],
            rescued=summary["rescued_requests"],
            harmed=summary["harmed_requests"],
            net=summary["net_failures_avoided"],
            linked=summary["harm_with_prior_relocation_out"],
            link_rate=summary["prior_out_link_rate"],
            within=summary["harm_within_60_minutes_of_prior_out"],
            within_rate=summary["within_60_minutes_rate_of_harm"],
            unlinked=summary["harm_without_prior_relocation_out"],
            median_minutes=summary["median_minutes_since_prior_out"] or 0,
        )
        for policy_name, summary in (
            ("greedy_default", default),
            ("greedy_service", service),
        )
    )
    top_rows = "\n".join(
        "| {label} | {station_id} {station_name} | {harm:,} | {net:+,} |".format(
            label=experiment.policies[policy_name]["label"],
            station_id=row["station_id"],
            station_name=row["station_name"],
            harm=row["harm_requests"],
            net=row["net_failures_avoided_vs_p0"],
        )
        for policy_name, summary in (
            ("greedy_default", default),
            ("greedy_service", service),
        )
        for row in summary["top_harm_stations"]
    )
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    return f"""# 강남구 P2 요청 단위 악화·구제 추적

## 결론

- 기존 P2: P0 실패→P2 성공 {default["rescued_requests"]:,}건,
  P0 성공→P2 실패 {default["harmed_requests"]:,}건,
  순 실패 방지 {default["net_failures_avoided"]:,}건
- 서비스 P2: P0 실패→P2 성공 {service["rescued_requests"]:,}건,
  P0 성공→P2 실패 {service["harmed_requests"]:,}건,
  순 실패 방지 {service["net_failures_avoided"]:,}건
- 악화 요청 중 같은 대여소의 선행 재배치 유출 연결률: 기존 P2
  {default["prior_out_link_rate"]:.2%}, 서비스 P2 {service["prior_out_link_rate"]:.2%}
- 악화 요청 중 선행 유출 60분 이내 비율: 기존 P2
  {default["within_60_minutes_rate_of_harm"]:.2%}, 서비스 P2
  {service["within_60_minutes_rate_of_harm"]:.2%}
- 순악화 대여소에서 발생한 악화 요청: 기존 P2
  {default["harm_at_net_worsened_stations"]:,}건, 서비스 P2
  {service["harm_at_net_worsened_stations"]:,}건

순개선은 `구제 요청 - 악화 요청`으로 재조정된다. 악화 요청 수는 순악화 건수보다 클 수
있으며, 같은 대여소나 다른 대여소의 구제 요청이 이를 상쇄한다.

## 실험 계약

- 방법: `{experiment.method}`
- 홀드아웃: `{experiment.evaluation_window}`
- 동일 요청 키: 원본 정제 데이터의 `trip_id`
- 구제: P0에서는 실패하고 P2에서는 성공한 같은 요청
- 악화: P0에서는 성공하고 P2에서는 실패한 같은 요청
- 선행 유출: 악화 요청 시각보다 이른 같은 대여소의 가장 최근 `relocation_out`
- 60분 기준: 재배치 판단 주기와 동일한 운영 구간
- 기본 시뮬레이션에서는 trace를 수집하지 않고 이 진단 실행에서만 활성화

## 정책별 요청 전환과 선행 유출

| 정책 | 구제 요청 | 악화 요청 | 순 방지 | 선행 유출 연결 | 60분 이내 | 미연결 | \
유출 후 중앙시간(분) |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

## 악화 요청 상위 대여소

`순 실패 증감`이 양수면 그 대여소 전체로는 개선됐다는 뜻이다. 즉 수혜 대여소에서도
일부 개별 요청은 P2 때문에 실패할 수 있다.

| 정책 | 대여소 | 악화 요청 | 대여소 순 실패 감소 |
|---|---|---:|---:|
{top_rows}

## 불변조건

- 세 정책의 rental trace는 같은 trip ID·요청 시각·대여소를 사용한다.
- 구제 요청 - 악화 요청 = 정책 전체 순 실패 방지다.
- 기본 trace 비활성 실행과 활성 실행의 정책 전체 지표가 같다.
- 성공 + 실패 = 전체 요청이며 자전거 보존식 잔차는 0이다.

## 해석 제한

{limitations}

## 다음 단계

1. 유출 직후보다 장시간 후 악화가 많다는 결과를 반영해 donor reserve 후보를 학습기간에서 비교한다.
2. P3는 전체 성공률과 악화 요청 수를 가중합하지 않고 두 지표를 별도 제약으로 둔다.
3. 후보를 고정한 뒤 홀드아웃에서 P2 대비 순개선·악화 요청·p10을 함께 검증한다.
"""


def _render_station_equity_markdown(experiment: StationEquityExperiment) -> str:
    default = experiment.equity_summaries["greedy_default"]
    service = experiment.equity_summaries["greedy_service"]
    p0 = next(row for row in experiment.policy_runs if row["policy_name"] == "no_relocation")
    summary_rows = "\n".join(
        "| {label} | {service_rate:.2%} | {p10:.2%} | {p10_delta:+.3f}%p | "
        "{improved}/{tied}/{worsened} | {net:+,} | {harm:,} | {concentration:.2%} |".format(
            label=experiment.policies[policy_name]["label"],
            service_rate=summary["overall_service_rate"],
            p10=summary["p10_station_service_rate"],
            p10_delta=summary["p10_service_rate_pp_vs_p0"],
            improved=summary["stations_improved"],
            tied=summary["stations_tied"],
            worsened=summary["stations_worsened"],
            net=summary["net_failures_avoided"],
            harm=summary["failures_added_at_worsened_stations"],
            concentration=summary["top_10_positive_concentration"],
        )
        for policy_name, summary in (
            ("greedy_default", default),
            ("greedy_service", service),
        )
    )

    def beneficiary_rows(policy_name: str, summary: dict[str, Any]) -> str:
        return "\n".join(
            "| {policy} | {station_id} {station_name} | {requests:,} | {avoided:+,} | "
            "{relocated_in:,} | {relocated_out:,} |".format(
                policy=experiment.policies[policy_name]["label"],
                station_id=row["station_id"],
                station_name=row["station_name"],
                requests=row["requests"],
                avoided=row["failures_avoided"],
                relocated_in=row["relocated_in"],
                relocated_out=row["relocated_out"],
            )
            for row in summary["top_10_beneficiary_stations"]
        )

    top_rows = "\n".join(
        (
            beneficiary_rows("greedy_default", default),
            beneficiary_rows("greedy_service", service),
        )
    )

    def worsened_rows(policy_name: str, summary: dict[str, Any]) -> str:
        rows = summary["worsened_stations"][:10]
        if not rows:
            return f"| {experiment.policies[policy_name]['label']} | 없음 | - | - | - | - |"
        return "\n".join(
            "| {policy} | {station_id} {station_name} | {requests:,} | {added:,} | "
            "{relocated_in:,} | {relocated_out:,} |".format(
                policy=experiment.policies[policy_name]["label"],
                station_id=row["station_id"],
                station_name=row["station_name"],
                requests=row["requests"],
                added=row["failures_added"],
                relocated_in=row["relocated_in"],
                relocated_out=row["relocated_out"],
            )
            for row in rows
        )

    harm_rows = "\n".join(
        (
            worsened_rows("greedy_default", default),
            worsened_rows("greedy_service", service),
        )
    )
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    return f"""# 강남구 P2 대여소별 공간 형평성 분석

## 결론

- 동일 대여소: {experiment.stations}개, 요청 발생 대여소:
  {default["stations_with_requests"]}개
- P0 전체 성공률 {p0["service_rate"]:.2%}, 요청 10건 이상 대여소 p10
  {p0["p10_station_service_rate"]:.2%}
- 기존 P2: 순 실패 방지 {default["net_failures_avoided"]:,}건, 개선/동률/악화
  {default["stations_improved"]}/{default["stations_tied"]}/{default["stations_worsened"]}개
- 서비스 P2: 순 실패 방지 {service["net_failures_avoided"]:,}건, 개선/동률/악화
  {service["stations_improved"]}/{service["stations_tied"]}/{service["stations_worsened"]}개
- 기존 P2 상위 10개 수혜 집중도: {default["top_10_positive_concentration"]:.2%}
- 서비스 P2 상위 10개 수혜 집중도: {service["top_10_positive_concentration"]:.2%}
- 재배치 유출이 있으면서 악화된 대여소: 기존 P2
  {default["relocation_out_and_worsened_stations"]}개, 서비스 P2
  {service["relocation_out_and_worsened_stations"]}개

전체 성공률만 보지 않고 악화 대여소와 하위 10% 서비스율을 함께 본다. 집중도는 각 정책의
모든 양의 실패 감소 중 상위 10개 대여소가 차지한 비율이며 순개선 대비 비율이 아니다.

## 실험 계약

- 방법: `{experiment.method}`
- 홀드아웃: `{experiment.evaluation_window}`
- 매일 리셋하지 않는 5일 연속 재생
- 동일 165개 대여소, 초기 재고, 관측 요청 순서
- P0: 재배치 없음
- 기존 P2: 시간당 직접 운송 2회, 15km/h, 적재 20대
- 서비스 P2: 시간당 직접 운송 3회, 20km/h, 적재 10대
- 개선·악화 기준: 같은 대여소 P0 대비 실패 건수 감소·증가
- p10 대상: 요청이 10건 이상인 대여소

## 정책별 형평성 요약

| 정책 | 전체 성공률 | 대여소 p10 | P0 대비 p10 | 개선/동률/악화 | 순 방지 실패 | \
악화지점 추가 실패 | 상위10 집중도 |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

## 실패 감소 상위 10개 대여소

| 정책 | 대여소 | 요청 | 방지 실패 | 재배치 유입 | 재배치 유출 |
|---|---|---:|---:|---:|---:|
{top_rows}

## 악화 대여소 상위 10개

전체 악화 대여소 목록과 수치는 CSV·JSON에 보존했다.

| 정책 | 대여소 | 요청 | 추가 실패 | 재배치 유입 | 재배치 유출 |
|---|---|---:|---:|---:|---:|
{harm_rows}

## 불변조건

- 세 정책의 대여소별 요청 수가 모두 같다.
- 대여소별 실패 합은 정책 전체 실패와 같다.
- 성공 + 실패 = 전체 요청이며 보존식 잔차는 0이다.
- 정책별 판단 시점 작업 수와 이동 대수 한도를 지킨다.

## 해석 제한

{limitations}

## 다음 단계

1. 악화 대여소의 시간대별 재고·수요·재배치 유출 순서를 분해한다.
2. 공급 대여소 보호 하한이나 악화 페널티가 있는 P3 후보를 설계한다.
3. P3는 동일 홀드아웃에서 사후 선택하지 않고 별도 검증기간 계약을 먼저 고정한다.
"""


def _render_temporal_markdown(experiment: TemporalRobustnessExperiment) -> str:
    summary_rows = "\n".join(
        "| {label} | {day_type} | {days} | {requests:,} | {service:.2%} | "
        "{failed:,} | {empty:,.1f} | {avoided:+,} | {better}/{tie}/{worse} |".format(
            label=row["policy_label"],
            day_type=row["day_type"],
            days=row["days"],
            requests=row["observed_requests"],
            service=row["weighted_service_rate"],
            failed=row["failed_rentals"],
            empty=row["empty_station_hours"],
            avoided=row["failures_avoided_vs_p0"],
            better=row["days_better_than_p0"],
            tie=row["days_tied_with_p0"],
            worse=row["days_worse_than_p0"],
        )
        for row in experiment.group_summaries
    )
    daily_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for row in experiment.daily_runs:
        daily_by_date.setdefault(row["date"], {})[row["policy_name"]] = row
    daily_rows = "\n".join(
        "| {date} | {day_type} | {requests:,} | {p0_failed:,} | {default_avoided:+,} | "
        "{service_avoided:+,} | {service_extra:+,} | {default_rate:.2%} | "
        "{service_rate:.2%} |".format(
            date=date,
            day_type=policies["no_relocation"]["day_type"],
            requests=policies["no_relocation"]["observed_requests"],
            p0_failed=policies["no_relocation"]["failed_rentals"],
            default_avoided=policies["greedy_default"]["failures_avoided_vs_p0"],
            service_avoided=policies["greedy_service"]["failures_avoided_vs_p0"],
            service_extra=policies["greedy_service"]["additional_failures_avoided_vs_default"],
            default_rate=policies["greedy_default"]["service_rate"],
            service_rate=policies["greedy_service"]["service_rate"],
        )
        for date, policies in sorted(daily_by_date.items())
    )

    def total_for(policy_name: str) -> dict[str, Any]:
        rows = [row for row in experiment.group_summaries if row["policy_name"] == policy_name]
        requests = sum(row["observed_requests"] for row in rows)
        successes = sum(row["successful_rentals"] for row in rows)
        return {
            "requests": requests,
            "failed": sum(row["failed_rentals"] for row in rows),
            "service_rate": _rate(successes, requests),
            "empty_hours": sum(row["empty_station_hours"] for row in rows),
            "failures_avoided": sum(row["failures_avoided_vs_p0"] for row in rows),
        }

    baseline = total_for("no_relocation")
    default = total_for("greedy_default")
    service = total_for("greedy_service")
    default_consistency = experiment.effect_consistency["greedy_default"]
    service_consistency = experiment.effect_consistency["greedy_service"]
    incremental_consistency = experiment.effect_consistency["greedy_service_vs_default"]
    excluded = (
        "없음"
        if not experiment.excluded_days
        else ", ".join(row["date"] for row in experiment.excluded_days)
    )
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    return f"""# 강남구 P2 날짜별 시간적 강건성 분석

## 결론

- 유효 일수: {experiment.valid_days}일, 제외일: {excluded}
- 월 전체 관측 요청: {baseline["requests"]:,}건
- P0: 성공률 {baseline["service_rate"]:.2%}, 실패 {baseline["failed"]:,}건
- 기존 P2(2회·15km/h): 성공률 {default["service_rate"]:.2%}, 실패
  {default["failed"]:,}건, P0 대비 {default["failures_avoided"]:,}건 방지
- 서비스 P2(3회·20km/h): 성공률 {service["service_rate"]:.2%}, 실패
  {service["failed"]:,}건, P0 대비 {service["failures_avoided"]:,}건 방지
- 기존 P2 일별 개선: {default_consistency["days_better_than_p0"]}/{experiment.valid_days}일,
  역전 {default_consistency["days_worse_than_p0"]}일, 일별 방지 실패
  {default_consistency["minimum_daily_failures_avoided"]}~
  {default_consistency["maximum_daily_failures_avoided"]}건
- 서비스 P2 일별 개선: {service_consistency["days_better_than_p0"]}/{experiment.valid_days}일,
  역전 {service_consistency["days_worse_than_p0"]}일, 일별 방지 실패
  {service_consistency["minimum_daily_failures_avoided"]}~
  {service_consistency["maximum_daily_failures_avoided"]}건
- 서비스 P2는 기존 P2 대비 {incremental_consistency["days_better_than_default"]}일 개선,
  {incremental_consistency["days_worse_than_default"]}일 악화했으며 악화일은
  {", ".join(incremental_consistency["worse_dates"]) or "없음"}

이 결과는 날짜별 효과 방향의 일관성을 보여주지만, 임계값 학습기간을 포함하므로 월 전체를
새로운 홀드아웃 검증으로 해석하지 않는다.

날짜별 P0는 매일 실제 자정 재고로 리셋된다. 따라서 이 보고서의 P0 성공률과 이전의 5일
연속 무재배치 P0는 서로 다른 초기화 계약이므로 수치를 직접 비교할 수 없다.

## 실험 계약

- 방법: `{experiment.method}`
- 분석 구간: `{experiment.analysis_window}`
- 동일 대상: 좌표와 자정 재고가 있는 {experiment.stations}개 대여소
- 일별 초기화: 매일 00:00 관측 재고에서 독립 시작
- P0: 재배치 없음
- 기존 P2: 시간당 직접 운송 2회, 15km/h, 적재 20대
- 서비스 P2: 시간당 직접 운송 3회, 20km/h, 적재 10대
- 공통: 임계값 2/5/8, 거리보정 1.3, 상하차 자전거당 0.75분

## 주중·주말 집계

개선일/동률일/악화일은 같은 날짜 P0의 실패 건수와 비교한 값이다.

| 정책 | 구분 | 일수 | 요청 | 성공률 | 실패 | 빈 시간 | 방지 실패 | 개선/동률/악화일 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

## 일별 결과

| 날짜 | 구분 | 요청 | P0 실패 | 기존 P2 방지 | 서비스 P2 방지 | 서비스 추가 | \
기존 성공률 | 서비스 성공률 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{daily_rows}

## 불변조건

- 각 날짜의 세 정책은 같은 {experiment.stations}개 대여소, 초기 재고, 요청 순서를 쓴다.
- 성공 + 실패 = 전체 요청이며 재배치 이동 중 자전거를 포함한 보존식 잔차는 0이다.
- 정책별 직접 운송 수·적재량 한도를 날짜마다 지킨다.

## 해석 제한

{limitations}

## 다음 단계

1. 대여소별 실패 감소가 소수 지역에 집중되는지 공간적 형평성을 비교한다.
2. 성과가 낮은 날짜와 대여소의 시간대별 실패 원인을 분해한다.
3. 그 결과를 바탕으로 수요예측 기반 P3의 입력·평가 계약을 설계한다.
"""


def _render_sensitivity_markdown(experiment: SensitivityExperiment) -> str:
    rows = "\n".join(
        "| {actions} | {capacity} | {speed:.0f} | {service:.2%} | {failed:,} | "
        "{empty:,.1f} | {distance:,.1f} | {minutes:,.1f} | {efficiency:,.1f} |".format(
            actions=row["max_actions_per_decision"],
            capacity=row["vehicle_capacity"],
            speed=row["average_speed_kmh"],
            service=row["service_rate"],
            failed=row["failed_rentals"],
            empty=row["empty_station_hours"],
            distance=row["relocation_distance_km"],
            minutes=row["relocation_vehicle_minutes"],
            efficiency=row["failures_avoided_per_100km"],
        )
        for row in experiment.runs
    )
    marginal_rows = "\n".join(
        "| {speed:.0f} | {capacity} | {before}→{after} | {avoided:+,} | "
        "{distance:+,.1f} | {minutes:+,.1f} | {efficiency:,.1f} |".format(
            speed=row["average_speed_kmh"],
            capacity=row["vehicle_capacity"],
            before=row["from_actions"],
            after=row["to_actions"],
            avoided=row["additional_failures_avoided"],
            distance=row["additional_distance_km"],
            minutes=row["additional_vehicle_minutes"],
            efficiency=row["additional_failures_avoided_per_100km"],
        )
        for row in experiment.marginal_actions
    )
    service = experiment.service_best
    empty_best = experiment.empty_time_best
    efficiency = experiment.distance_efficiency_best
    reference = experiment.default_scenario
    baseline = experiment.baseline
    factors = experiment.factors
    findings = experiment.factor_findings
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    failed_values = [row["failed_rentals"] for row in experiment.runs]
    service_values = [row["service_rate"] for row in experiment.runs]
    empty_values = [row["empty_station_hours"] for row in experiment.runs]
    return f"""# 강남구 P2 공간 재배치 민감도 분석

## 결론

- 18개 운영 조합의 성공률 범위: {min(service_values):.2%}~{max(service_values):.2%}
- 실패 범위: {min(failed_values):,}~{max(failed_values):,}건
- 빈 대여소 누적 시간 범위: {min(empty_values):,.1f}~{max(empty_values):,.1f}시간
- 서비스 최상: 시간당 {service["max_actions_per_decision"]}회, 적재
  {service["equivalent_vehicle_capacities"]}대, {service["average_speed_kmh"]:.0f}km/h
  (성공률 {service["service_rate"]:.2%}, 실패 {service["failed_rentals"]:,}건)
- 빈 시간 최소: 시간당 {empty_best["max_actions_per_decision"]}회, 적재
  {empty_best["equivalent_vehicle_capacities"]}대,
  {empty_best["average_speed_kmh"]:.0f}km/h
  ({empty_best["empty_station_hours"]:,.1f}시간, 실패 {empty_best["failed_rentals"]:,}건)
- 거리 효율 최상: 시간당 {efficiency["max_actions_per_decision"]}회, 적재
  {efficiency["vehicle_capacity"]}대, {efficiency["average_speed_kmh"]:.0f}km/h
  (100km당 방지 실패 {efficiency["failures_avoided_per_100km"]:,.1f}건)
- 기존 P2 기준값(2회·20대·15km/h): 성공률 {reference["service_rate"]:.2%},
  실패 {reference["failed_rentals"]:,}건

서비스 최상과 거리 효율 최상은 별도로 제시했다. 인건비·차량비 단가가 없으므로 임의의
가중 종합점수나 ‘최적 차량 수’는 산출하지 않는다.

## 요인별 발견

- 시간당 운송 횟수에 따른 동일 속도·적재 내 최대 실패 차이:
  {findings["max_failed_rental_spread_from_actions"]:,}건
- 속도에 따른 동일 운송 횟수·적재 내 최대 실패 차이:
  {findings["max_failed_rental_spread_from_speed"]:,}건
- 적재량에 따른 동일 운송 횟수·속도 내 최대 실패 차이:
  {findings["max_failed_rental_spread_from_capacity"]:,}건
- 적재량 제약이 작동하지 않은 이유: {findings["capacity_non_binding_reason"]}

따라서 이 실험 범위에서는 적재량보다 시간당 직접 운송 횟수가 성과 변동을 더 크게
설명한다. 다만 이는 인과적 요인 중요도가 아니라 선택한 18개 조합 안의 관측 범위다.

## 실험 계약

- 방법: `{experiment.method}`
- 검증 구간: `{experiment.evaluation_window}`
- 동일 비교 대여소: {experiment.stations}개
- 좌표 미확보 제외 ID: {", ".join(experiment.excluded_station_ids) or "없음"}
- P0 기준: 요청 {baseline["observed_requests"]:,}건, 성공률
  {baseline["service_rate"]:.2%}, 실패 {baseline["failed_rentals"]:,}건
- 판단 주기: {factors["decision_interval_minutes"]}분
- 시간당 직접 운송: {factors["max_direct_trips_per_decision"]}
- 평균속도(km/h): {factors["average_speed_kmh"]}
- 차량 적재량(대): {factors["vehicle_capacity"]}
- 조합 전체를 수용하는 시나리오 이동 상한: 시간당
  {factors["scenario_max_bikes_per_decision"]}대
- 임계값 하한·목표·상한: {factors["lower_target_upper_thresholds"]}
- 거리: Haversine 직선거리 × {factors["road_distance_factor"]}
- 상하차: 자전거당 {factors["handling_minutes_per_bike"]:.2f}분

## 18개 조합 결과

| 시간당 운송 | 적재 | 속도 | 성공률 | 실패 | 빈 시간 | 거리(km) | 차량분 | 100km당 방지 실패 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## 직접 운송 횟수 추가의 한계효과

양수 ‘추가 방지 실패’는 운송 횟수 증가로 실패가 감소했다는 뜻이다. 이 표는 동일 속도와
적재량 안에서만 비교한다.

| 속도 | 적재 | 시간당 운송 | 추가 방지 실패 | 추가 거리(km) | 추가 차량분 | \
추가 100km당 방지 실패 |
|---:|---:|---:|---:|---:|---:|---:|
{marginal_rows}

## 불변조건

- 모든 조합은 같은 {experiment.stations}개 대여소, 초기 재고, 관측 요청 순서를 사용한다.
- 성공 + 실패 = 전체 요청이며 재배치 중 자전거를 포함한 보존식 잔차는 0이다.
- 각 조합의 판단 시점별 작업 수와 작업당 적재량은 해당 요인값을 넘지 않는다.

## 해석 제한

{limitations}

## 다음 단계

1. 날짜별로 P0와 대표 P2 조합을 재실행해 특정 요일에 치우친 결과인지 확인한다.
2. 일별 효과 분포와 부트스트랩 신뢰구간으로 시간적 강건성을 측정한다.
3. 그 뒤에만 비용 단가를 입력받아 운영비 대비 서비스 개선을 비교한다.
"""


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


def _render_spatial_markdown(experiment: SpatialSimulationExperiment) -> str:
    rows = "\n".join(
        "| {label} | {requests:,} | {service:.2%} | {failed:,} | {empty:,.1f} | "
        "{moved:,} | {actions:,} | {distance:,.1f} | {minutes:,.1f} |".format(
            label=row["policy_label"],
            requests=row["observed_requests"],
            service=row["service_rate"],
            failed=row["failed_rentals"],
            empty=row["empty_station_hours"],
            moved=row["bikes_moved"],
            actions=row["relocation_actions"],
            distance=row["relocation_distance_km"],
            minutes=row["relocation_vehicle_minutes"],
        )
        for row in experiment.evaluation_runs
    )
    assumptions = experiment.assumptions
    p2 = experiment.comparisons["greedy_nearest"]
    limitations = "\n".join(f"- {item}" for item in experiment.limitations)
    return f"""# 강남구 공간 재배치 P0/P1/P2 비교

## 결론

- P2가 P0 대비 방지한 관측 대여 실패: {p2["failures_avoided_vs_p0"]:,.0f}건
- P2의 P0 대비 성공률 개선: {p2["service_rate_percentage_point_vs_p0"]:.3f}%p
- P2의 P0 대비 빈 대여소 시간 감소: {p2["empty_hours_reduced_vs_p0"]:,.1f}시간 \
({p2["empty_time_reduction_rate_vs_p0"]:.2%})
- 이동 지연 때문에 P1보다 추가된 실패: {p2["additional_failures_vs_p1"]:,.0f}건

P1은 이동을 즉시 완료하는 낙관적 상한이고, P2는 거리와 직접 운송시간을 적용한 두 번째 \
운영 근사다. 두 결과 모두 실제 현장 인과효과가 아니다.

## 실험 계약

- 방법: `{experiment.method}`
- 검증 구간: `{experiment.evaluation_window}`
- 동일 비교 대여소: {experiment.stations}개
- 좌표 미확보 제외 ID: {", ".join(experiment.excluded_station_ids) or "없음"}
- 좌표 파일: `{experiment.coordinate_file}`
- 판단 주기: {assumptions["decision_interval_minutes"]}분
- P1 판단 시점별 이동 한도: {assumptions["max_bikes_per_decision"]}대, 직접 운송 횟수 미적용
- P2 판단 시점별 한도: 최대 {assumptions["max_direct_trips_per_decision"]}회, \
전체 {assumptions["max_bikes_per_decision"]}대
- 차량 한 대 적재량: {assumptions["vehicle_capacity"]}대
- 평균속도: {assumptions["average_speed_kmh"]:.1f}km/h
- 거리: Haversine 직선거리 × {assumptions["road_distance_factor"]:.1f}
- 상하차: 자전거당 {assumptions["handling_minutes_per_bike"]:.2f}분
- P1/P2 임계값: 하한 {assumptions["lower_threshold"]}, \
목표 {assumptions["target_bikes"]}, 상한 {assumptions["upper_threshold"]}

## 홀드아웃 결과

| 정책 | 요청 | 성공률 | 실패 | 빈 시간 | 이동 대수 | 직접 운송 | 거리(km) | 차량분 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## 불변조건

- 세 정책은 같은 {experiment.stations}개 대여소, 초기 재고, 관측 요청 순서를 사용한다.
- P2 재배치 자전거는 출발 즉시 공급 대여소에서 빠지고 이동시간 후 목적지에 도착한다.
- 성공 + 실패 = 전체 요청이며 재고는 음수가 되지 않는다.
- 사용자 이동 중 자전거와 재배치 이동 중 자전거를 포함한 보존식 잔차는 0이다.

## 해석 제한

{limitations}

## 다음 단계

1. 차량의 현재 위치와 공급 대여소까지의 첫 접근 이동을 추가한다.
2. 차량 수·속도·적재량 민감도 분석으로 결과 범위를 제시한다.
3. 관측되지 않은 잠재 수요를 낮음·기준·높음 시나리오로 확장한다.
"""


def _plot_harm_trace(
    frame: pl.DataFrame,
    summaries: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_names = ("greedy_default", "greedy_service")
    policy_labels = ("기존 P2", "서비스 P2")
    colors = ("#1f77b4", "#ff7f0e")
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    with plt.rc_context({"font.family": font_family, "axes.unicode_minus": False}):
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        x_values = list(range(len(policy_names)))
        bar_width = 0.24
        rescued = [summaries[name]["rescued_requests"] for name in policy_names]
        harmed = [summaries[name]["harmed_requests"] for name in policy_names]
        net = [summaries[name]["net_failures_avoided"] for name in policy_names]
        axes[0, 0].bar(
            [value - bar_width for value in x_values],
            rescued,
            width=bar_width,
            label="구제",
            color="#59a14f",
        )
        axes[0, 0].bar(
            x_values,
            harmed,
            width=bar_width,
            label="악화",
            color="#e15759",
        )
        axes[0, 0].bar(
            [value + bar_width for value in x_values],
            net,
            width=bar_width,
            label="순 방지",
            color="#4e79a7",
        )
        axes[0, 0].set_xticks(x_values, policy_labels)
        axes[0, 0].set_title("동일 요청의 결과 전환")
        axes[0, 0].set_ylabel("요청(건)")
        axes[0, 0].grid(axis="y", alpha=0.2)
        axes[0, 0].legend()

        within = [summaries[name]["harm_within_60_minutes_of_prior_out"] for name in policy_names]
        linked_after = [
            summaries[name]["harm_with_prior_relocation_out"] - within[index]
            for index, name in enumerate(policy_names)
        ]
        unlinked = [summaries[name]["harm_without_prior_relocation_out"] for name in policy_names]
        axes[0, 1].bar(policy_labels, within, label="유출 후 60분 이내", color="#f28e2b")
        axes[0, 1].bar(
            policy_labels,
            linked_after,
            bottom=within,
            label="유출 후 60분 초과",
            color="#edc948",
        )
        axes[0, 1].bar(
            policy_labels,
            unlinked,
            bottom=[first + second for first, second in zip(within, linked_after, strict=True)],
            label="선행 유출 없음",
            color="#bab0ac",
        )
        axes[0, 1].set_title("악화 요청과 가장 최근 선행 유출")
        axes[0, 1].set_ylabel("악화 요청(건)")
        axes[0, 1].grid(axis="y", alpha=0.2)
        axes[0, 1].legend()

        bin_edges = (0, 60, 180, 360, 720, 1_440, math.inf)
        bin_labels = ("≤60", "1~3h", "3~6h", "6~12h", "12~24h", ">24h")
        for index, (policy_name, label, color) in enumerate(
            zip(policy_names, policy_labels, colors, strict=True)
        ):
            minutes = frame.filter(
                (pl.col("policy_name") == policy_name)
                & pl.col("minutes_since_prior_out").is_not_null()
            )["minutes_since_prior_out"].to_list()
            counts = [
                sum(lower < value <= upper for value in minutes)
                if lower > 0
                else sum(lower <= value <= upper for value in minutes)
                for lower, upper in zip(bin_edges, bin_edges[1:], strict=False)
            ]
            positions = [value + (index - 0.5) * 0.36 for value in range(len(bin_labels))]
            axes[1, 0].bar(positions, counts, width=0.36, label=label, color=color)
        axes[1, 0].set_xticks(range(len(bin_labels)), bin_labels)
        axes[1, 0].set_title("악화 요청까지 선행 유출 경과시간")
        axes[1, 0].set_xlabel("가장 최근 유출 후 경과")
        axes[1, 0].set_ylabel("악화 요청(건)")
        axes[1, 0].grid(axis="y", alpha=0.2)
        axes[1, 0].legend()

        station_counts: dict[tuple[str, str], dict[str, int]] = {}
        for row in frame.iter_rows(named=True):
            key = (str(row["station_id"]), str(row["station_name"]))
            station_counts.setdefault(key, dict.fromkeys(policy_names, 0))
            station_counts[key][str(row["policy_name"])] += 1
        top_stations = sorted(
            station_counts,
            key=lambda key: (-sum(station_counts[key].values()), key[0]),
        )[:10]
        y_values = list(range(len(top_stations)))
        for index, (policy_name, label, color) in enumerate(
            zip(policy_names, policy_labels, colors, strict=True)
        ):
            positions = [value + (index - 0.5) * 0.36 for value in y_values]
            axes[1, 1].barh(
                positions,
                [station_counts[key][policy_name] for key in top_stations],
                height=0.36,
                label=label,
                color=color,
            )
        axes[1, 1].set_yticks(
            y_values,
            [f"{station_id} {station_name}" for station_id, station_name in top_stations],
        )
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("악화 요청 상위 대여소")
        axes[1, 1].set_xlabel("악화 요청(건)")
        axes[1, 1].grid(axis="x", alpha=0.2)
        axes[1, 1].legend()
        figure.suptitle("강남구 2025-11 홀드아웃: 요청 단위 악화·구제 추적")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def _plot_station_equity(
    frame: pl.DataFrame,
    runs: tuple[tuple[str, SimulationRun], ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    with plt.rc_context({"font.family": font_family, "axes.unicode_minus": False}):
        figure, axes = plt.subplots(2, 2, figsize=(15, 11))
        for axis, (prefix, title) in zip(
            axes[0],
            (
                ("default", "기존 P2: 대여소별 P0 대비 실패 감소"),
                ("service", "서비스 P2: 대여소별 P0 대비 실패 감소"),
            ),
            strict=True,
        ):
            values = frame[f"{prefix}_failures_avoided_vs_p0"].to_list()
            limit = max(max(abs(value) for value in values), 1)
            scatter = axis.scatter(
                frame["longitude"].to_list(),
                frame["latitude"].to_list(),
                c=values,
                cmap="RdYlGn",
                vmin=-limit,
                vmax=limit,
                s=[20 + math.sqrt(requests) * 2 for requests in frame["requests"].to_list()],
                alpha=0.85,
                edgecolors="#333333",
                linewidths=0.2,
            )
            axis.set_title(title)
            axis.set_xlabel("경도")
            axis.set_ylabel("위도")
            axis.grid(alpha=0.15)
            colorbar = figure.colorbar(scatter, ax=axis, shrink=0.82)
            colorbar.set_label("방지 실패(건, 음수는 악화)")

        labels = [label for label, _ in runs]
        overall = [run.metrics.service_rate * 100 for _, run in runs]
        p10 = [run.metrics.p10_station_service_rate * 100 for _, run in runs]
        worst = [run.metrics.worst_station_service_rate * 100 for _, run in runs]
        x_values = list(range(len(labels)))
        bar_width = 0.24
        axes[1, 0].bar(
            [value - bar_width for value in x_values],
            overall,
            width=bar_width,
            label="전체",
            color="#4c78a8",
        )
        axes[1, 0].bar(
            x_values,
            p10,
            width=bar_width,
            label="대여소 p10",
            color="#f2cf5b",
        )
        axes[1, 0].bar(
            [value + bar_width for value in x_values],
            worst,
            width=bar_width,
            label="최저 대여소",
            color="#e45756",
        )
        axes[1, 0].set_xticks(x_values, labels, rotation=8)
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].set_title("전체와 하위 대여소 서비스율")
        axes[1, 0].set_ylabel("성공률(%)")
        axes[1, 0].grid(axis="y", alpha=0.2)
        axes[1, 0].legend()

        active = frame.filter(pl.col("requests") > 0)
        policy_labels = [runs[1][0], runs[2][0]]
        improved = []
        tied = []
        worsened = []
        for prefix in ("default", "service"):
            values = active[f"{prefix}_failures_avoided_vs_p0"]
            improved.append(int((values > 0).sum()))
            tied.append(int((values == 0).sum()))
            worsened.append(int((values < 0).sum()))
        axes[1, 1].bar(policy_labels, improved, label="개선", color="#59a14f")
        axes[1, 1].bar(policy_labels, tied, bottom=improved, label="동률", color="#bab0ac")
        axes[1, 1].bar(
            policy_labels,
            worsened,
            bottom=[good + same for good, same in zip(improved, tied, strict=True)],
            label="악화",
            color="#e15759",
        )
        axes[1, 1].set_title("요청 발생 대여소의 개선·동률·악화")
        axes[1, 1].set_ylabel("대여소 수")
        axes[1, 1].tick_params(axis="x", rotation=8)
        axes[1, 1].grid(axis="y", alpha=0.2)
        axes[1, 1].legend()
        figure.suptitle("강남구 2025-11 홀드아웃: P2 공간 형평성")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def _plot_temporal_robustness(
    frame: pl.DataFrame,
    group_summaries: tuple[dict[str, Any], ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = sorted(frame["date"].unique().to_list())
    x_values = list(range(len(dates)))
    tick_positions = list(range(0, len(dates), 4))
    tick_labels = [dates[index][5:] for index in tick_positions]
    policy_rows = (
        frame.select("policy_order", "policy_name", "policy_label").unique().sort("policy_order")
    )
    colors = {
        "no_relocation": "#7f7f7f",
        "greedy_default": "#1f77b4",
        "greedy_service": "#ff7f0e",
    }
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    with plt.rc_context({"font.family": font_family}):
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        for policy in policy_rows.iter_rows(named=True):
            subset = frame.filter(pl.col("policy_name") == policy["policy_name"]).sort("date")
            axes[0, 0].plot(
                x_values,
                subset["failed_rentals"].to_list(),
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=colors[policy["policy_name"]],
                label=policy["policy_label"],
            )
            axes[1, 1].plot(
                x_values,
                subset["empty_station_hours"].to_list(),
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=colors[policy["policy_name"]],
                label=policy["policy_label"],
            )
        for policy_name in ("greedy_default", "greedy_service"):
            subset = frame.filter(pl.col("policy_name") == policy_name).sort("date")
            label = subset["policy_label"][0]
            axes[0, 1].plot(
                x_values,
                subset["failures_avoided_vs_p0"].to_list(),
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=colors[policy_name],
                label=label,
            )
        for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
            axis.set_xticks(tick_positions, tick_labels, rotation=35)
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0, 0].set_title("날짜별 대여 실패")
        axes[0, 0].set_ylabel("실패(건)")
        axes[0, 1].axhline(0, color="#333333", linewidth=1)
        axes[0, 1].set_title("P0 대비 날짜별 방지 실패")
        axes[0, 1].set_ylabel("방지 실패(건)")
        axes[1, 1].set_title("날짜별 빈 대여소 누적 시간")
        axes[1, 1].set_ylabel("대여소·시간")

        summary_map = {(row["policy_name"], row["day_type"]): row for row in group_summaries}
        day_types = ("주중", "주말")
        bar_width = 0.24
        for index, policy in enumerate(policy_rows.iter_rows(named=True)):
            values = [
                summary_map[(policy["policy_name"], day_type)]["weighted_service_rate"] * 100
                for day_type in day_types
            ]
            positions = [value + (index - 1) * bar_width for value in range(len(day_types))]
            axes[1, 0].bar(
                positions,
                values,
                width=bar_width,
                color=colors[policy["policy_name"]],
                label=policy["policy_label"],
            )
        axes[1, 0].set_xticks(range(len(day_types)), day_types)
        axes[1, 0].set_ylim(80, 100)
        axes[1, 0].set_title("주중·주말 가중 성공률")
        axes[1, 0].set_ylabel("성공률(%)")
        axes[1, 0].grid(axis="y", alpha=0.2)
        axes[1, 0].legend()
        figure.suptitle("강남구 2025-11 P2 날짜별 시간적 강건성")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def _plot_spatial_sensitivity(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    capacities = sorted(frame["vehicle_capacity"].unique().to_list())
    speeds = sorted(frame["average_speed_kmh"].unique().to_list())
    panel_count = len(capacities) + 2
    columns = 2
    rows = math.ceil(panel_count / columns)
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    colors = plt.get_cmap("viridis").resampled(max(len(speeds), 2))
    with plt.rc_context({"font.family": font_family}):
        figure, axes = plt.subplots(rows, columns, figsize=(14, 5.2 * rows), squeeze=False)
        flat_axes = axes.flat
        for capacity, axis in zip(capacities, flat_axes, strict=False):
            for index, speed in enumerate(speeds):
                subset = frame.filter(
                    (pl.col("vehicle_capacity") == capacity)
                    & (pl.col("average_speed_kmh") == speed)
                ).sort("max_actions_per_decision")
                axis.plot(
                    subset["max_actions_per_decision"].to_list(),
                    subset["failed_rentals"].to_list(),
                    marker="o",
                    linewidth=2,
                    color=colors(index),
                    label=f"{speed:g}km/h",
                )
            axis.set_title(f"적재 {capacity}대: 시간당 운송과 실패")
            axis.set_xlabel("시간당 최대 직접 운송(회)")
            axis.set_ylabel("실패 대여(건)")
            axis.set_xticks(sorted(frame["max_actions_per_decision"].unique().to_list()))
            axis.grid(alpha=0.2)
            axis.legend()

        distance_axis = axes.flat[len(capacities)]
        action_values = frame["max_actions_per_decision"].to_list()
        scatter = distance_axis.scatter(
            frame["relocation_distance_km"].to_list(),
            frame["failed_rentals"].to_list(),
            c=action_values,
            cmap="plasma",
            s=[45 + capacity * 2 for capacity in frame["vehicle_capacity"].to_list()],
            alpha=0.8,
        )
        distance_axis.set_title("서비스-거리 상충관계")
        distance_axis.set_xlabel("재배치 거리(km)")
        distance_axis.set_ylabel("실패 대여(건, 낮을수록 좋음)")
        distance_axis.grid(alpha=0.2)
        colorbar = figure.colorbar(scatter, ax=distance_axis)
        colorbar.set_label("시간당 직접 운송(회)")

        time_axis = axes.flat[len(capacities) + 1]
        time_axis.scatter(
            frame["relocation_vehicle_minutes"].to_list(),
            frame["failed_rentals"].to_list(),
            c=action_values,
            cmap="plasma",
            s=[45 + capacity * 2 for capacity in frame["vehicle_capacity"].to_list()],
            alpha=0.8,
        )
        time_axis.set_title("서비스-차량시간 상충관계")
        time_axis.set_xlabel("차량시간(분)")
        time_axis.set_ylabel("실패 대여(건, 낮을수록 좋음)")
        time_axis.grid(alpha=0.2)
        for axis in axes.flat[panel_count:]:
            axis.set_visible(False)
        figure.suptitle("강남구 P2 공간 재배치 운영 가정 민감도")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def _plot_spatial_comparison(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = frame["policy_label"].to_list()
    panels = (
        ("failures_per_1000_requests", "1,000건당 대여 실패", "건"),
        ("empty_station_hours", "빈 대여소 누적 시간", "대여소·시간"),
        ("bikes_moved", "재배치 작업량", "대"),
        ("relocation_distance_km", "직접 운송 거리", "km"),
    )
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = "DejaVu Sans"
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        font_family = font_manager.FontProperties(fname=str(malgun_path)).get_name()
    with plt.rc_context({"font.family": font_family}):
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e"]
        for axis, (column, title, unit) in zip(axes.flat, panels, strict=True):
            values = frame[column].to_list()
            bars = axis.bar(labels, values, color=colors)
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.2)
            axis.tick_params(axis="x", rotation=8)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:,.1f}",
                    ha="center",
                    va="bottom",
                )
        figure.suptitle("강남구 2025-11 홀드아웃: 공간 재배치 정책 비교")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


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


def _haversine_km(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> float:
    latitude_1, longitude_1 = map(math.radians, origin)
    latitude_2, longitude_2 = map(math.radians, destination)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2) ** 2
    )
    return 6_371.0088 * 2 * math.asin(math.sqrt(haversine))
