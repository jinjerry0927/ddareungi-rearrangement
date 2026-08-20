from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import font_manager

from ddareungi_rearrangement.historical_audit import StationInfo, normalize_station_id

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RENTAL_FIELDS = (
    "자전거번호",
    "대여일시",
    "대여 대여소번호",
    "대여 대여소명",
    "대여거치대",
    "반납일시",
    "반납대여소번호",
    "반납대여소명",
    "반납거치대",
    "이용시간(분)",
    "이용거리(M)",
    "생년",
    "성별",
    "이용자종류",
    "대여대여소ID",
    "반납대여소ID",
    "자전거구분",
)
INVENTORY_FIELDS = ("일시", "대여소번호", "대여소명", "시간대", "거치대수량")

TRIP_SCHEMA = pa.schema(
    [
        ("rent_at", pa.timestamp("us")),
        ("return_at", pa.timestamp("us")),
        ("rent_station_id", pa.string()),
        ("rent_station_name", pa.string()),
        ("return_station_id", pa.string()),
        ("return_station_name", pa.string()),
        ("duration_minutes", pa.int32()),
        ("distance_m", pa.float64()),
        ("bike_type", pa.string()),
        ("rent_in_scope", pa.bool_()),
        ("return_in_scope", pa.bool_()),
        ("flow_type", pa.string()),
    ]
)
INVENTORY_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("us")),
        ("station_id", pa.string()),
        ("station_name", pa.string()),
        ("borough", pa.string()),
        ("available_bikes", pa.int32()),
        ("nominal_racks", pa.int32()),
        ("is_empty", pa.bool_()),
        ("over_nominal", pa.bool_()),
    ]
)


class PilotAnalysisError(RuntimeError):
    """파일 구조 또는 품질이 파일럿 분석 계약을 위반할 때 발생하는 오류."""


@dataclass(frozen=True)
class TripExtractionAudit:
    source_member: str
    source_rows: int
    scoped_rows: int
    flow_counts: dict[str, int]
    invalid_rent_datetimes: int
    missing_return_datetimes: int
    invalid_duration_values: int
    invalid_distance_values: int
    output_file: str


@dataclass(frozen=True)
class InventoryExtractionAudit:
    source_member: str
    source_rows: int
    scoped_rows: int
    unique_station_ids: int
    invalid_rows: int
    output_file: str


@dataclass(frozen=True)
class PilotBaseline:
    generated_at_utc: str
    borough: str
    month: str
    stations: int
    active_stations: int
    actionable_stations: int
    no_activity_station_ids: tuple[str, ...]
    station_hours: int
    inventory_observed_hours: int
    inventory_coverage_rate: float
    raw_empty_station_hours: int
    raw_empty_station_hour_rate: float
    active_inventory_observed_hours: int
    empty_station_hours: int
    empty_station_hour_rate: float
    over_nominal_station_hours: int
    over_nominal_rate: float
    scoped_trips: int
    internal_trips: int
    outbound_trips: int
    inbound_trips: int
    unresolved_return_trips: int
    rental_events: int
    return_events: int
    return_events_outside_month: int
    peak_empty_hour: int
    peak_empty_hour_rate: float
    busiest_rental_hour: int
    busiest_return_hour: int
    highest_shortage_station_id: str
    highest_shortage_station_name: str
    highest_shortage_rate: float
    largest_net_outflow_station: str
    largest_net_outflow: int
    largest_net_inflow_station: str
    largest_net_inflow: int
    top_shortage_stations: tuple[dict[str, Any], ...]
    trip_extraction: TripExtractionAudit
    inventory_extraction: InventoryExtractionAudit
    output_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_borough_trips(
    rental_zip: Path,
    stations: dict[str, StationInfo],
    *,
    month: str,
    borough: str,
    output_path: Path,
    batch_size: int = 50_000,
) -> TripExtractionAudit:
    member_suffix = f"_{month[2:4]}{month[5:7]}.csv"
    source_rows = 0
    scoped_rows = 0
    invalid_rent_datetimes = 0
    missing_return_datetimes = 0
    invalid_duration_values = 0
    invalid_distance_values = 0
    flow_counts: Counter[str] = Counter()
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        archive = zipfile.ZipFile(rental_zip)
    except (OSError, zipfile.BadZipFile):
        raise PilotAnalysisError("대여이력 ZIP을 열지 못했습니다") from None

    with archive:
        member = _find_member(archive, member_suffix)
        try:
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="cp949", newline="") as text:
                    reader = csv.DictReader(text)
                    if tuple(reader.fieldnames or ()) != RENTAL_FIELDS:
                        raise PilotAnalysisError("11월 대여이력 CSV 헤더가 변경됐습니다")

                    for row in reader:
                        source_rows += 1
                        rent_station_id = normalize_station_id(row["대여 대여소번호"])
                        return_station_id = normalize_station_id(row["반납대여소번호"])
                        rent_info = stations.get(rent_station_id)
                        return_info = stations.get(return_station_id)
                        rent_in_scope = bool(rent_info and rent_info.borough == borough)
                        return_in_scope = bool(return_info and return_info.borough == borough)
                        if not rent_in_scope and not return_in_scope:
                            continue

                        rent_at = _parse_datetime(row["대여일시"])
                        if rent_at is None:
                            invalid_rent_datetimes += 1
                            continue
                        return_at = _parse_datetime(row["반납일시"])
                        if return_at is None:
                            missing_return_datetimes += 1

                        duration = _parse_nonnegative_int(row["이용시간(분)"])
                        if duration is None:
                            invalid_duration_values += 1
                        distance = _parse_nonnegative_float(row["이용거리(M)"])
                        if distance is None:
                            invalid_distance_values += 1

                        if return_at is None:
                            flow_type = "unresolved_return"
                        elif rent_in_scope and return_in_scope:
                            flow_type = "internal"
                        elif rent_in_scope:
                            flow_type = "outbound"
                        else:
                            flow_type = "inbound"

                        batch.append(
                            {
                                "rent_at": rent_at,
                                "return_at": return_at,
                                "rent_station_id": rent_station_id,
                                "rent_station_name": row["대여 대여소명"].strip(),
                                "return_station_id": return_station_id or None,
                                "return_station_name": row["반납대여소명"].strip() or None,
                                "duration_minutes": duration,
                                "distance_m": distance,
                                "bike_type": row["자전거구분"].strip(),
                                "rent_in_scope": rent_in_scope,
                                "return_in_scope": return_in_scope,
                                "flow_type": flow_type,
                            }
                        )
                        scoped_rows += 1
                        flow_counts[flow_type] += 1
                        if len(batch) >= batch_size:
                            writer = _write_batch(output_path, TRIP_SCHEMA, batch, writer)
                            batch.clear()
        finally:
            if batch:
                writer = _write_batch(output_path, TRIP_SCHEMA, batch, writer)
            if writer is not None:
                writer.close()

    if scoped_rows == 0:
        raise PilotAnalysisError("강남구에 연결된 11월 대여이력이 없습니다")

    return TripExtractionAudit(
        source_member=member.filename,
        source_rows=source_rows,
        scoped_rows=scoped_rows,
        flow_counts=dict(sorted(flow_counts.items())),
        invalid_rent_datetimes=invalid_rent_datetimes,
        missing_return_datetimes=missing_return_datetimes,
        invalid_duration_values=invalid_duration_values,
        invalid_distance_values=invalid_distance_values,
        output_file=str(output_path),
    )


def extract_borough_inventory(
    inventory_zip: Path,
    stations: dict[str, StationInfo],
    *,
    month: str,
    borough: str,
    output_path: Path,
    batch_size: int = 50_000,
) -> InventoryExtractionAudit:
    member_suffix = f"_{month[2:4]}{month[5:7]}.csv"
    source_rows = 0
    scoped_rows = 0
    invalid_rows = 0
    station_ids: set[str] = set()
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        archive = zipfile.ZipFile(inventory_zip)
    except (OSError, zipfile.BadZipFile):
        raise PilotAnalysisError("재고 스냅샷 ZIP을 열지 못했습니다") from None

    with archive:
        member = _find_member(archive, member_suffix)
        try:
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="cp949", newline="") as text:
                    reader = csv.DictReader(text)
                    if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
                        raise PilotAnalysisError("11월 재고 CSV 헤더가 변경됐습니다")

                    for row in reader:
                        source_rows += 1
                        station_id = normalize_station_id(row["대여소번호"])
                        info = stations.get(station_id)
                        if not info or info.borough != borough:
                            continue
                        try:
                            timestamp = datetime.fromisoformat(
                                f"{row['일시'].strip()} {int(row['시간대']):02d}:00:00"
                            )
                            available_bikes = int(row["거치대수량"])
                        except ValueError:
                            invalid_rows += 1
                            continue

                        batch.append(
                            {
                                "timestamp": timestamp,
                                "station_id": station_id,
                                "station_name": info.station_name,
                                "borough": info.borough,
                                "available_bikes": available_bikes,
                                "nominal_racks": info.nominal_racks,
                                "is_empty": available_bikes == 0,
                                "over_nominal": available_bikes > info.nominal_racks,
                            }
                        )
                        scoped_rows += 1
                        station_ids.add(station_id)
                        if len(batch) >= batch_size:
                            writer = _write_batch(output_path, INVENTORY_SCHEMA, batch, writer)
                            batch.clear()
        finally:
            if batch:
                writer = _write_batch(output_path, INVENTORY_SCHEMA, batch, writer)
            if writer is not None:
                writer.close()

    if scoped_rows == 0:
        raise PilotAnalysisError("강남구 11월 재고 스냅샷이 없습니다")

    return InventoryExtractionAudit(
        source_member=member.filename,
        source_rows=source_rows,
        scoped_rows=scoped_rows,
        unique_station_ids=len(station_ids),
        invalid_rows=invalid_rows,
        output_file=str(output_path),
    )


def build_pilot_baseline(
    stations: dict[str, StationInfo],
    *,
    borough: str,
    month: str,
    trip_audit: TripExtractionAudit,
    inventory_audit: InventoryExtractionAudit,
    trips_path: Path,
    inventory_path: Path,
    station_hour_path: Path,
    hourly_csv_path: Path,
    station_csv_path: Path,
    hourly_figure_path: Path,
    station_figure_path: Path,
) -> PilotBaseline:
    scoped_stations = sorted(
        (info for info in stations.values() if info.borough == borough),
        key=lambda info: int(info.station_id) if info.station_id.isdigit() else info.station_id,
    )
    station_frame = pl.DataFrame(
        {
            "station_id": [info.station_id for info in scoped_stations],
            "station_name": [info.station_name for info in scoped_stations],
            "borough": [info.borough for info in scoped_stations],
            "nominal_racks": [info.nominal_racks for info in scoped_stations],
        }
    )
    month_start = datetime.fromisoformat(f"{month}-01 00:00:00")
    month_end = month_start.replace(day=_days_in_month(month), hour=23)
    hours = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                month_start,
                month_end,
                interval="1h",
                eager=True,
            )
        }
    )
    grid = hours.join(station_frame, how="cross")

    trips = pl.read_parquet(trips_path)
    rental_events = (
        trips.filter(pl.col("rent_in_scope"))
        .with_columns(pl.col("rent_at").dt.truncate("1h").alias("timestamp"))
        .group_by("timestamp", pl.col("rent_station_id").alias("station_id"))
        .agg(pl.len().alias("rentals"))
    )
    return_events = (
        trips.filter(pl.col("return_in_scope") & pl.col("return_at").is_not_null())
        .with_columns(pl.col("return_at").dt.truncate("1h").alias("timestamp"))
        .group_by("timestamp", pl.col("return_station_id").alias("station_id"))
        .agg(pl.len().alias("returns"))
    )
    flow = rental_events.join(
        return_events,
        on=["timestamp", "station_id"],
        how="full",
        coalesce=True,
    ).with_columns(
        pl.col("rentals").fill_null(0).cast(pl.Int64),
        pl.col("returns").fill_null(0).cast(pl.Int64),
    )

    inventory = pl.read_parquet(inventory_path).select(
        "timestamp",
        "station_id",
        "available_bikes",
        "is_empty",
        "over_nominal",
    )
    station_hour = (
        grid.join(inventory, on=["timestamp", "station_id"], how="left")
        .join(flow, on=["timestamp", "station_id"], how="left")
        .with_columns(
            pl.col("available_bikes").is_not_null().alias("inventory_observed"),
            pl.col("rentals").fill_null(0),
            pl.col("returns").fill_null(0),
        )
        .with_columns((pl.col("returns") - pl.col("rentals")).alias("net_flow"))
        .sort("timestamp", "station_id")
    )
    minimum_observed_hours = int(hours.height * 0.8)
    station_ranking = (
        station_hour.group_by("station_id", "station_name")
        .agg(
            pl.col("inventory_observed").sum().alias("observed_hours"),
            pl.col("is_empty").fill_null(False).sum().alias("empty_hours"),
            pl.col("over_nominal").fill_null(False).sum().alias("over_nominal_hours"),
            pl.col("available_bikes").max().fill_null(0).alias("max_available_bikes"),
            pl.col("rentals").sum(),
            pl.col("returns").sum(),
            pl.col("net_flow").sum(),
        )
        .with_columns(
            (pl.col("rentals") + pl.col("returns")).alias("total_events"),
            (pl.col("empty_hours") / pl.col("observed_hours")).alias("empty_rate"),
        )
        .with_columns(
            ((pl.col("max_available_bikes") > 0) | (pl.col("total_events") > 0)).alias(
                "activity_evidence"
            ),
            (
                (pl.col("total_events") >= 100)
                & (pl.col("observed_hours") >= minimum_observed_hours)
            ).alias("actionable"),
        )
        .sort("actionable", "empty_rate", "rentals", descending=True)
    )
    station_hour = station_hour.join(
        station_ranking.select("station_id", "activity_evidence", "actionable"),
        on="station_id",
        how="left",
    )
    station_hour_path.parent.mkdir(parents=True, exist_ok=True)
    station_hour.write_parquet(station_hour_path, compression="zstd")

    hourly = (
        station_hour.filter(pl.col("activity_evidence"))
        .with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.col("rentals").sum(),
            pl.col("returns").sum(),
            pl.col("inventory_observed").sum().alias("observed_station_hours"),
            pl.col("is_empty").fill_null(False).sum().alias("empty_station_hours"),
            pl.col("available_bikes").mean().alias("average_available_bikes"),
        )
        .with_columns(
            (pl.col("empty_station_hours") / pl.col("observed_station_hours")).alias("empty_rate")
        )
        .sort("hour")
    )
    hourly_csv_path.parent.mkdir(parents=True, exist_ok=True)
    station_csv_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_csv(hourly_csv_path)
    station_ranking.write_csv(station_csv_path)
    _plot_hourly(hourly, hourly_figure_path, borough, month)
    _plot_station_ranking(station_ranking, station_figure_path, borough, month)

    observed_hours = int(station_hour["inventory_observed"].sum())
    raw_empty_hours = int(station_hour["is_empty"].fill_null(False).sum())
    active_station_hour = station_hour.filter(pl.col("activity_evidence"))
    active_observed_hours = int(active_station_hour["inventory_observed"].sum())
    empty_hours = int(active_station_hour["is_empty"].fill_null(False).sum())
    over_nominal_hours = int(station_hour["over_nominal"].fill_null(False).sum())
    peak_empty = hourly.sort("empty_rate", descending=True).row(0, named=True)
    busiest_rental = hourly.sort("rentals", descending=True).row(0, named=True)
    busiest_return = hourly.sort("returns", descending=True).row(0, named=True)
    actionable_ranking = station_ranking.filter(pl.col("actionable"))
    if actionable_ranking.is_empty():
        actionable_ranking = station_ranking.filter(pl.col("activity_evidence"))
    top_station = actionable_ranking.row(0, named=True)
    net_outflow_station = actionable_ranking.sort("net_flow").row(0, named=True)
    net_inflow_station = actionable_ranking.sort("net_flow", descending=True).row(0, named=True)
    top_shortage_stations = tuple(
        {
            "station_id": str(row["station_id"]),
            "station_name": str(row["station_name"]),
            "empty_rate": float(row["empty_rate"]),
            "rentals": int(row["rentals"]),
            "returns": int(row["returns"]),
            "net_flow": int(row["net_flow"]),
        }
        for row in actionable_ranking.head(10).iter_rows(named=True)
    )
    no_activity_ids = tuple(
        sorted(
            station_ranking.filter(~pl.col("activity_evidence"))["station_id"].to_list(),
            key=lambda value: int(value) if value.isdigit() else value,
        )
    )
    flow_counts = trip_audit.flow_counts

    return PilotBaseline(
        generated_at_utc=datetime.now(UTC).isoformat(),
        borough=borough,
        month=month,
        stations=len(scoped_stations),
        active_stations=int(station_ranking["activity_evidence"].sum()),
        actionable_stations=int(station_ranking["actionable"].sum()),
        no_activity_station_ids=no_activity_ids,
        station_hours=station_hour.height,
        inventory_observed_hours=observed_hours,
        inventory_coverage_rate=_rate(observed_hours, station_hour.height),
        raw_empty_station_hours=raw_empty_hours,
        raw_empty_station_hour_rate=_rate(raw_empty_hours, observed_hours),
        active_inventory_observed_hours=active_observed_hours,
        empty_station_hours=empty_hours,
        empty_station_hour_rate=_rate(empty_hours, active_observed_hours),
        over_nominal_station_hours=over_nominal_hours,
        over_nominal_rate=_rate(over_nominal_hours, observed_hours),
        scoped_trips=trip_audit.scoped_rows,
        internal_trips=flow_counts.get("internal", 0),
        outbound_trips=flow_counts.get("outbound", 0),
        inbound_trips=flow_counts.get("inbound", 0),
        unresolved_return_trips=flow_counts.get("unresolved_return", 0),
        rental_events=int(station_hour["rentals"].sum()),
        return_events=int(station_hour["returns"].sum()),
        return_events_outside_month=(
            int(trips.filter(pl.col("return_in_scope") & pl.col("return_at").is_not_null()).height)
            - int(station_hour["returns"].sum())
        ),
        peak_empty_hour=int(peak_empty["hour"]),
        peak_empty_hour_rate=float(peak_empty["empty_rate"]),
        busiest_rental_hour=int(busiest_rental["hour"]),
        busiest_return_hour=int(busiest_return["hour"]),
        highest_shortage_station_id=str(top_station["station_id"]),
        highest_shortage_station_name=str(top_station["station_name"]),
        highest_shortage_rate=float(top_station["empty_rate"]),
        largest_net_outflow_station=(
            f"{net_outflow_station['station_id']}. {net_outflow_station['station_name']}"
        ),
        largest_net_outflow=int(net_outflow_station["net_flow"]),
        largest_net_inflow_station=(
            f"{net_inflow_station['station_id']}. {net_inflow_station['station_name']}"
        ),
        largest_net_inflow=int(net_inflow_station["net_flow"]),
        top_shortage_stations=top_shortage_stations,
        trip_extraction=trip_audit,
        inventory_extraction=inventory_audit,
        output_files={
            "trips": str(trips_path),
            "inventory": str(inventory_path),
            "station_hour": str(station_hour_path),
            "hourly_profile": str(hourly_csv_path),
            "station_ranking": str(station_csv_path),
            "hourly_figure": str(hourly_figure_path),
            "station_figure": str(station_figure_path),
        },
    )


def write_pilot_reports(
    baseline: PilotBaseline,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_pilot_markdown(baseline), encoding="utf-8")


def render_pilot_markdown(baseline: PilotBaseline) -> str:
    flow_total = (
        baseline.internal_trips
        + baseline.outbound_trips
        + baseline.inbound_trips
        + baseline.unresolved_return_trips
    )
    empty_summary = (
        f"{baseline.empty_station_hours:,} / {baseline.active_inventory_observed_hours:,} "
        f"({baseline.empty_station_hour_rate:.2%})"
    )
    raw_empty_summary = (
        f"{baseline.raw_empty_station_hours:,} / {baseline.inventory_observed_hours:,} "
        f"({baseline.raw_empty_station_hour_rate:.2%})"
    )
    no_activity_ids = ", ".join(baseline.no_activity_station_ids)
    no_activity_count = len(baseline.no_activity_station_ids)
    top_station_summary = (
        f"{baseline.highest_shortage_station_id}. {baseline.highest_shortage_station_name} "
        f"({baseline.highest_shortage_rate:.2%})"
    )
    internal_row = (
        f"| 강남구 내부 | {baseline.internal_trips:,} | "
        f"{_rate(baseline.internal_trips, flow_total):.2%} |"
    )
    outbound_row = (
        f"| 강남구 출발·외부 도착 | {baseline.outbound_trips:,} | "
        f"{_rate(baseline.outbound_trips, flow_total):.2%} |"
    )
    inbound_row = (
        f"| 외부 출발·강남구 도착 | {baseline.inbound_trips:,} | "
        f"{_rate(baseline.inbound_trips, flow_total):.2%} |"
    )
    unresolved_row = (
        f"| 반납 미확정 | {baseline.unresolved_return_trips:,} | "
        f"{_rate(baseline.unresolved_return_trips, flow_total):.2%} |"
    )
    shortage_rows = "\n".join(
        f"| {index} | {row['station_id']} | {row['station_name']} | "
        f"{row['empty_rate']:.2%} | {row['rentals']:,} | {row['returns']:,} | "
        f"{row['net_flow']:+,} |"
        for index, row in enumerate(baseline.top_shortage_stations, start=1)
    )
    return f"""# {baseline.month} {baseline.borough} 재배치 전 기준선

- 생성시각(UTC): {baseline.generated_at_utc}
- 범위: {baseline.borough}에서 출발하거나 도착한 전체 이동
- 메타데이터 대여소 / 활동 증거 대여소: {baseline.stations:,} / {baseline.active_stations:,}개
- 월 100건 이상 분석 대상 대여소: {baseline.actionable_stations:,}개
- 범위 내 이동 레코드: {baseline.scoped_trips:,}건

## 핵심 결과

- 재고가 0대인 대여소-시간: {empty_summary}
- 활동 여부 구분 전 원시 0대 비율: {raw_empty_summary}
- 재고 관측 커버리지: {baseline.inventory_coverage_rate:.2%}
- 명목 거치대 수 초과: {baseline.over_nominal_station_hours:,} ({baseline.over_nominal_rate:.2%})
- 0대 비율 최고 시간대: {baseline.peak_empty_hour:02d}시 ({baseline.peak_empty_hour_rate:.2%})
- 대여량 최고 시간대: {baseline.busiest_rental_hour:02d}시
- 반납량 최고 시간대: {baseline.busiest_return_hour:02d}시
- 0대 비율 최고 대여소: {top_station_summary}
- 월 순유출 최대: {baseline.largest_net_outflow_station} ({baseline.largest_net_outflow:+,}대)
- 월 순유입 최대: {baseline.largest_net_inflow_station} ({baseline.largest_net_inflow:+,}대)

한 달 내 재고가 한 번도 0보다 크지 않고 대여·반납도 없는 대여소 {no_activity_count}개는
운영 증거 없음으로 분리했다: `{no_activity_ids}`. 최고 부족 대여소 순위는 관측시간 80% 이상,
월 대여·반납 합계 100건 이상인 대여소만 비교했다.

## 부족 관측 상위 대여소

| 순위 | ID | 대여소 | 0대 비율 | 대여 | 반납 | 순유입 |
|---:|---:|---|---:|---:|---:|---:|
{shortage_rows}

순유입은 `반납-대여`이며 음수일수록 재배치가 없을 때 자전거가 빠져나가는 방향이다.

## 이동 경계

| 유형 | 건수 | 비율 |
|---|---:|---:|
{internal_row}
{outbound_row}
{inbound_row}
{unresolved_row}

강남구 내부 이동만 사용하면 외부에서 들어오고 나가는 자전거 흐름이 사라지므로, 파일럿은
강남구를 한쪽 끝점으로 갖는 이동을 모두 유지했다.

- 11월 강남구 대여 이벤트: {baseline.rental_events:,}
- 11월 강남구 반납 이벤트: {baseline.return_events:,}
- 12월에 반납되어 11월 시간 집계에서 제외된 이벤트: {baseline.return_events_outside_month:,}

## 데이터 품질

- 11월 서울 전체 원천 행: {baseline.trip_extraction.source_rows:,}
- 범위 내 정제 행: {baseline.trip_extraction.scoped_rows:,}
- 대여일시 오류로 제외: {baseline.trip_extraction.invalid_rent_datetimes:,}
- 반납일시 결측: {baseline.trip_extraction.missing_return_datetimes:,}
- 이용시간 변환 오류: {baseline.trip_extraction.invalid_duration_values:,}
- 이용거리 변환 오류: {baseline.trip_extraction.invalid_distance_values:,}
- 강남구 재고 행: {baseline.inventory_extraction.scoped_rows:,}
- 재고 변환 오류: {baseline.inventory_extraction.invalid_rows:,}

자전거번호·생년·성별·이용자종류는 재배치 기준선에 필요하지 않아 정제 Parquet에서 제외했다.

## 해석 범위

이 기준선의 `0대`는 관측 시각에 대여 가능한 자전거가 없었다는 뜻이며 실제 대여 실패 건수는
아니다. 또한 명목 거치대 수는 하드 용량이 아니므로 초과 관측을 반납 실패로 해석하지 않는다.
다음 시뮬레이션은 우선 `0대 대여소-시간 감소`를 1차 목적함수로 사용해야 한다.
대여이력 파일은 대여 월 기준이므로 10월 말에 대여해 11월 초 반납한 이동은 포함되지 않는다.
시뮬레이션 학습 구간은 월 경계 영향을 피하도록 11월 3일부터 28일까지 사용한다.
"""


def _find_member(archive: zipfile.ZipFile, suffix: str) -> zipfile.ZipInfo:
    matches = [entry for entry in archive.infolist() if entry.filename.endswith(suffix)]
    if len(matches) != 1:
        raise PilotAnalysisError(f"ZIP에서 대상 월 CSV를 하나로 찾지 못했습니다: {suffix}")
    return matches[0]


def _write_batch(
    output_path: Path,
    schema: pa.Schema,
    rows: list[dict[str, Any]],
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    writer.write_table(table)
    return writer


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_nonnegative_int(value: str) -> int | None:
    try:
        result = int(float(value.strip()))
    except ValueError:
        return None
    return result if result >= 0 else None


def _parse_nonnegative_float(value: str) -> float | None:
    try:
        result = float(value.strip())
    except ValueError:
        return None
    return result if result >= 0 else None


def _days_in_month(month: str) -> int:
    year, month_number = (int(part) for part in month.split("-"))
    if month_number == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month_number + 1, 1)
    this_month = datetime(year, month_number, 1)
    return (next_month - this_month).days


def _plot_hourly(frame: pl.DataFrame, path: Path, borough: str, month: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = frame["hour"].to_list()
    with plt.rc_context({"font.family": _korean_font_family()}):
        fig, axis = plt.subplots(figsize=(11, 6))
        axis.plot(hours, frame["rentals"].to_list(), marker="o", label="Rentals")
        axis.plot(hours, frame["returns"].to_list(), marker="o", label="Returns")
        axis.set_xlabel("Hour of day")
        axis.set_ylabel("Monthly events")
        axis.set_xticks(range(24))
        axis.grid(alpha=0.2)
        second = axis.twinx()
        second.plot(
            hours,
            [value * 100 for value in frame["empty_rate"].to_list()],
            color="#d62728",
            linewidth=2.5,
            label="Empty station-hours",
        )
        second.set_ylabel("Empty station-hours (%)")
        lines = axis.get_lines() + second.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
        axis.set_title(f"{borough} {month}: hourly flow and empty-station baseline")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


def _plot_station_ranking(
    frame: pl.DataFrame,
    path: Path,
    borough: str,
    month: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eligible = frame.filter(pl.col("actionable"))
    if eligible.is_empty():
        eligible = frame.filter(pl.col("activity_evidence"))
    top = eligible.head(15).sort("empty_rate")
    labels = [
        f"{station_id} {station_name}"
        for station_id, station_name in zip(
            top["station_id"].to_list(),
            top["station_name"].to_list(),
            strict=True,
        )
    ]
    with plt.rc_context({"font.family": _korean_font_family()}):
        fig, axis = plt.subplots(figsize=(11, 7))
        axis.barh(labels, [value * 100 for value in top["empty_rate"].to_list()])
        axis.set_xlabel("Empty station-hours (%)")
        axis.set_title(f"{borough} {month}: stations with the highest empty rate")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


def _korean_font_family() -> str:
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    if malgun_path.exists():
        font_manager.fontManager.addfont(str(malgun_path))
        return font_manager.FontProperties(fname=str(malgun_path)).get_name()
    return "DejaVu Sans"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
