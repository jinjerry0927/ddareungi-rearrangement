from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ddareungi_rearrangement.seoul_api import RentalHistoryPage

STATION_SOURCE_URL = "https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do"
INVENTORY_SOURCE_URL = "https://data.seoul.go.kr/dataList/OA-22382/F/1/datasetView.do"
RENTAL_SOURCE_URL = "https://data.seoul.go.kr/dataList/OA-15182/A/1/datasetView.do"

INVENTORY_FIELDS = ("일시", "대여소번호", "대여소명", "시간대", "거치대수량")
RENTAL_REQUIRED_FIELDS = (
    "RENT_DT",
    "RENT_ID",
    "RENT_NM",
    "RTN_DT",
    "RTN_ID",
    "RTN_NM",
    "USE_MIN",
    "USE_DST",
    "RENT_STATION_ID",
    "RETURN_STATION_ID",
)


class HistoricalDataError(RuntimeError):
    """원천 데이터가 예상 계약을 만족하지 않을 때 발생하는 오류."""


@dataclass(frozen=True)
class StationInfo:
    station_id: str
    station_name: str
    borough: str
    nominal_racks: int


@dataclass(frozen=True)
class StationAudit:
    source_file: str
    sheet_name: str
    rows: int
    unique_station_ids: int
    duplicate_station_ids: int
    boroughs: int
    missing_station_ids: int
    missing_boroughs: int
    missing_nominal_racks: int
    lcd_station_rows: int
    qr_station_rows: int
    hybrid_station_rows: int
    nominal_racks_min: int
    nominal_racks_median: float
    nominal_racks_max: int


@dataclass(frozen=True)
class InventoryMonthAudit:
    member: str
    month: str
    rows: int
    dates: int
    hours: tuple[int, ...]
    time_groups: int
    missing_time_groups: tuple[str, ...]
    station_time_coverage_rate: float
    unique_station_ids: int
    matched_station_ids: int
    missing_values: int
    invalid_dates: int
    invalid_hours: int
    invalid_inventory_values: int
    negative_inventory_values: int
    duplicate_station_time_keys: int
    non_monotonic_time_groups: int
    zero_inventory_rows: int
    matched_capacity_rows: int
    inventory_over_nominal_racks_rows: int


@dataclass(frozen=True)
class BoroughInventoryAudit:
    borough: str
    rows: int
    unique_station_ids: int
    matched_capacity_rate: float
    zero_inventory_rows: int
    zero_inventory_rate: float
    inventory_over_nominal_racks_rows: int
    over_nominal_racks_rate: float
    peak_zero_hours: tuple[int, ...]


@dataclass(frozen=True)
class InventoryAudit:
    source_file: str
    encoding: str
    target_month: str
    total_rows: int
    unique_station_ids: int
    matched_station_ids: int
    station_mapping_rate: float
    zero_inventory_rows: int
    inventory_over_nominal_racks_rows: int
    months: tuple[InventoryMonthAudit, ...]
    borough_ranking: tuple[BoroughInventoryAudit, ...]
    candidate_borough: str
    candidate_peak_hours: tuple[int, ...]


@dataclass(frozen=True)
class RentalAudit:
    query: str
    response_service_names: tuple[str, ...]
    result_codes: tuple[str, ...]
    reported_count: int
    received_rows: int
    observed_fields: tuple[str, ...]
    field_types: dict[str, tuple[str, ...]]
    missing_values: dict[str, int]
    invalid_duration_values: int
    invalid_distance_values: int
    unique_rental_station_ids: int
    unique_return_station_ids: int
    rental_station_mapping_rate: float
    return_station_mapping_rate: float
    candidate_borough_rentals: int
    candidate_borough_returns: int


@dataclass(frozen=True)
class HistoricalAudit:
    audited_at_utc: str
    status: str
    analysis_ready: bool
    simulator_ready: bool
    selected_period: str
    candidate_borough: str
    candidate_peak_hours: tuple[int, ...]
    station: StationAudit
    inventory: InventoryAudit
    rental: RentalAudit
    warnings: tuple[str, ...]
    source_urls: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StationDataset:
    audit: StationAudit
    stations: dict[str, StationInfo]


def audit_station_workbook(path: Path) -> StationDataset:
    try:
        frame = pl.read_excel(
            path,
            sheet_name="대여소현황",
            has_header=False,
            drop_empty_rows=False,
        )
    except Exception as exc:  # pragma: no cover - engine-specific details are unstable
        raise HistoricalDataError(f"대여소 XLSX를 읽지 못했습니다: {type(exc).__name__}") from None

    return audit_station_frame(frame, source_file=path.name, sheet_name="대여소현황")


def audit_station_frame(
    frame: pl.DataFrame,
    *,
    source_file: str = "fixture.xlsx",
    sheet_name: str = "대여소현황",
) -> StationDataset:
    if frame.height < 6 or frame.width < 10:
        raise HistoricalDataError("대여소 XLSX의 행 또는 열 수가 예상보다 작습니다")

    rows = frame.rows()
    if _clean(rows[0][0]) != "대여소\n번호" or _clean(rows[2][2]) != "자치구":
        raise HistoricalDataError("대여소 XLSX의 다중 헤더 구조가 변경됐습니다")

    stations: dict[str, StationInfo] = {}
    duplicate_station_ids = 0
    missing_station_ids = 0
    missing_boroughs = 0
    missing_nominal_racks = 0
    lcd_station_rows = 0
    qr_station_rows = 0
    hybrid_station_rows = 0
    capacities: list[int] = []
    boroughs: set[str] = set()
    data_rows = 0

    for row in rows[5:]:
        if not any(_clean(value) for value in row[:10]):
            continue
        data_rows += 1
        station_id = normalize_station_id(row[0])
        station_name = _clean(row[1])
        borough = _clean(row[2])
        lcd_racks = _optional_nonnegative_int(row[7])
        qr_racks = _optional_nonnegative_int(row[8])

        if not station_id:
            missing_station_ids += 1
            continue
        if station_id in stations:
            duplicate_station_ids += 1
            continue
        if not borough:
            missing_boroughs += 1
        else:
            boroughs.add(borough)

        if lcd_racks is not None:
            lcd_station_rows += 1
        if qr_racks is not None:
            qr_station_rows += 1
        if lcd_racks is not None and qr_racks is not None:
            hybrid_station_rows += 1

        nominal_racks = (lcd_racks or 0) + (qr_racks or 0)
        if nominal_racks <= 0:
            missing_nominal_racks += 1
        else:
            capacities.append(nominal_racks)

        stations[station_id] = StationInfo(
            station_id=station_id,
            station_name=station_name,
            borough=borough,
            nominal_racks=nominal_racks,
        )

    if not capacities:
        raise HistoricalDataError("대여소 XLSX에서 유효한 거치대 수를 찾지 못했습니다")

    sorted_capacities = sorted(capacities)
    middle = len(sorted_capacities) // 2
    if len(sorted_capacities) % 2:
        median = float(sorted_capacities[middle])
    else:
        median = (sorted_capacities[middle - 1] + sorted_capacities[middle]) / 2

    audit = StationAudit(
        source_file=source_file,
        sheet_name=sheet_name,
        rows=data_rows,
        unique_station_ids=len(stations),
        duplicate_station_ids=duplicate_station_ids,
        boroughs=len(boroughs),
        missing_station_ids=missing_station_ids,
        missing_boroughs=missing_boroughs,
        missing_nominal_racks=missing_nominal_racks,
        lcd_station_rows=lcd_station_rows,
        qr_station_rows=qr_station_rows,
        hybrid_station_rows=hybrid_station_rows,
        nominal_racks_min=min(capacities),
        nominal_racks_median=median,
        nominal_racks_max=max(capacities),
    )
    return StationDataset(audit=audit, stations=stations)


def audit_inventory_zip(
    path: Path,
    stations: dict[str, StationInfo],
    *,
    target_month: str,
) -> InventoryAudit:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        raise HistoricalDataError("과거 재고 ZIP을 열지 못했습니다") from None

    month_audits: list[InventoryMonthAudit] = []
    all_station_ids: set[str] = set()
    matched_station_ids: set[str] = set()
    total_rows = 0
    zero_inventory_rows = 0
    over_nominal_rows = 0
    borough_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    borough_stations: dict[str, set[str]] = defaultdict(set)
    borough_hour_counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    with archive:
        members = sorted(
            (
                entry
                for entry in archive.infolist()
                if not entry.is_dir() and entry.filename.endswith(".csv")
            ),
            key=lambda entry: entry.filename,
        )
        if not members:
            raise HistoricalDataError("과거 재고 ZIP에 CSV가 없습니다")

        for member in members:
            month_audit, aggregates = _audit_inventory_member(
                archive,
                member,
                stations,
                target_month=target_month,
            )
            month_audits.append(month_audit)
            total_rows += month_audit.rows
            zero_inventory_rows += month_audit.zero_inventory_rows
            over_nominal_rows += month_audit.inventory_over_nominal_racks_rows
            all_station_ids.update(aggregates["all_station_ids"])
            matched_station_ids.update(aggregates["matched_station_ids"])
            for borough, counts in aggregates["borough_counts"].items():
                for index, value in enumerate(counts):
                    borough_counts[borough][index] += value
            for borough, station_ids in aggregates["borough_stations"].items():
                borough_stations[borough].update(station_ids)
            for key, counts in aggregates["borough_hour_counts"].items():
                borough_hour_counts[key][0] += counts[0]
                borough_hour_counts[key][1] += counts[1]

    ranking: list[BoroughInventoryAudit] = []
    for borough, counts in borough_counts.items():
        rows, matched_rows, zero_rows, over_rows = counts
        hour_rates = []
        for hour in range(6, 23):
            hour_rows, hour_zero = borough_hour_counts[(borough, hour)]
            if hour_rows:
                hour_rates.append((hour_zero / hour_rows, hour))
        peak_hours = tuple(hour for _, hour in sorted(hour_rates, reverse=True)[:3])
        ranking.append(
            BoroughInventoryAudit(
                borough=borough,
                rows=rows,
                unique_station_ids=len(borough_stations[borough]),
                matched_capacity_rate=_rate(matched_rows, rows),
                zero_inventory_rows=zero_rows,
                zero_inventory_rate=_rate(zero_rows, rows),
                inventory_over_nominal_racks_rows=over_rows,
                over_nominal_racks_rate=_rate(over_rows, matched_rows),
                peak_zero_hours=peak_hours,
            )
        )

    eligible = [
        row for row in ranking if row.unique_station_ids >= 75 and row.matched_capacity_rate >= 0.95
    ]
    ordered = sorted(
        ranking,
        key=lambda row: (row.zero_inventory_rate, row.unique_station_ids, row.borough),
        reverse=True,
    )
    eligible_ordered = sorted(
        eligible,
        key=lambda row: (row.zero_inventory_rate, row.unique_station_ids, row.borough),
        reverse=True,
    )
    candidate = eligible_ordered[0] if eligible_ordered else ordered[0]

    return InventoryAudit(
        source_file=path.name,
        encoding="cp949",
        target_month=target_month,
        total_rows=total_rows,
        unique_station_ids=len(all_station_ids),
        matched_station_ids=len(matched_station_ids),
        station_mapping_rate=_rate(len(matched_station_ids), len(all_station_ids)),
        zero_inventory_rows=zero_inventory_rows,
        inventory_over_nominal_racks_rows=over_nominal_rows,
        months=tuple(month_audits),
        borough_ranking=tuple(ordered),
        candidate_borough=candidate.borough,
        candidate_peak_hours=candidate.peak_zero_hours,
    )


def _audit_inventory_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    stations: dict[str, StationInfo],
    *,
    target_month: str,
) -> tuple[InventoryMonthAudit, dict[str, Any]]:
    rows = 0
    dates: set[str] = set()
    hours: set[int] = set()
    station_ids: set[str] = set()
    matched_ids: set[str] = set()
    missing_values = 0
    invalid_dates = 0
    invalid_hours = 0
    invalid_inventory = 0
    negative_inventory = 0
    duplicate_keys = 0
    non_monotonic_groups = 0
    zero_rows = 0
    matched_capacity_rows = 0
    over_nominal_rows = 0
    current_time_group: tuple[str, int] | None = None
    current_group_stations: set[str] = set()
    observed_time_groups: set[tuple[str, int]] = set()
    valid_dates: dict[str, bool] = {}
    observed_month = ""
    borough_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    borough_stations: dict[str, set[str]] = defaultdict(set)
    borough_hour_counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="cp949", newline="") as text:
            reader = csv.DictReader(text)
            if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
                raise HistoricalDataError(f"과거 재고 CSV 헤더가 변경됐습니다: {member.filename}")

            for row in reader:
                rows += 1
                if any(not _clean(row.get(field)) for field in INVENTORY_FIELDS):
                    missing_values += 1

                date_text = _clean(row.get("일시"))
                if date_text not in valid_dates:
                    try:
                        date.fromisoformat(date_text)
                    except ValueError:
                        valid_dates[date_text] = False
                    else:
                        valid_dates[date_text] = True
                if not valid_dates[date_text]:
                    invalid_dates += 1
                    continue
                dates.add(date_text)
                row_month = date_text[:7]
                observed_month = observed_month or row_month

                try:
                    hour = int(_clean(row.get("시간대")))
                except ValueError:
                    invalid_hours += 1
                    continue
                if not 0 <= hour <= 23:
                    invalid_hours += 1
                    continue
                hours.add(hour)

                try:
                    inventory = int(_clean(row.get("거치대수량")))
                except ValueError:
                    invalid_inventory += 1
                    continue
                if inventory < 0:
                    negative_inventory += 1

                station_id = normalize_station_id(row.get("대여소번호"))
                if not station_id:
                    continue
                station_ids.add(station_id)
                info = stations.get(station_id)
                if info:
                    matched_ids.add(station_id)

                time_group = (date_text, hour)
                if time_group != current_time_group:
                    if current_time_group is not None and time_group < current_time_group:
                        non_monotonic_groups += 1
                    current_time_group = time_group
                    current_group_stations.clear()
                    observed_time_groups.add(time_group)
                if station_id in current_group_stations:
                    duplicate_keys += 1
                current_group_stations.add(station_id)

                is_zero = inventory == 0
                if is_zero:
                    zero_rows += 1
                if info and info.nominal_racks > 0:
                    matched_capacity_rows += 1
                    if inventory > info.nominal_racks:
                        over_nominal_rows += 1

                if row_month == target_month and info and info.borough:
                    counts = borough_counts[info.borough]
                    counts[0] += 1
                    counts[1] += int(info.nominal_racks > 0)
                    counts[2] += int(is_zero)
                    counts[3] += int(info.nominal_racks > 0 and inventory > info.nominal_racks)
                    borough_stations[info.borough].add(station_id)
                    hour_counts = borough_hour_counts[(info.borough, hour)]
                    hour_counts[0] += 1
                    hour_counts[1] += int(is_zero)

    missing_time_groups = tuple(
        f"{date_text}/{hour:02d}"
        for date_text in sorted(dates)
        for hour in range(24)
        if (date_text, hour) not in observed_time_groups
    )
    time_groups = len(observed_time_groups)
    audit = InventoryMonthAudit(
        member=member.filename,
        month=observed_month,
        rows=rows,
        dates=len(dates),
        hours=tuple(sorted(hours)),
        time_groups=time_groups,
        missing_time_groups=missing_time_groups,
        station_time_coverage_rate=_rate(rows, len(station_ids) * time_groups),
        unique_station_ids=len(station_ids),
        matched_station_ids=len(matched_ids),
        missing_values=missing_values,
        invalid_dates=invalid_dates,
        invalid_hours=invalid_hours,
        invalid_inventory_values=invalid_inventory,
        negative_inventory_values=negative_inventory,
        duplicate_station_time_keys=duplicate_keys,
        non_monotonic_time_groups=non_monotonic_groups,
        zero_inventory_rows=zero_rows,
        matched_capacity_rows=matched_capacity_rows,
        inventory_over_nominal_racks_rows=over_nominal_rows,
    )
    return audit, {
        "all_station_ids": station_ids,
        "matched_station_ids": matched_ids,
        "borough_counts": borough_counts,
        "borough_stations": borough_stations,
        "borough_hour_counts": borough_hour_counts,
    }


def audit_rental_pages(
    pages: list[RentalHistoryPage],
    stations: dict[str, StationInfo],
    *,
    candidate_borough: str,
) -> RentalAudit:
    rows = [row for page in pages for row in page.rows]
    observed_fields = tuple(sorted({field for row in rows for field in row}))
    field_types = {
        field: tuple(sorted({type(row.get(field)).__name__ for row in rows if field in row}))
        for field in observed_fields
    }
    missing_values = {
        field: sum(not _clean(row.get(field)) for row in rows) for field in RENTAL_REQUIRED_FIELDS
    }
    rental_ids = {
        station_id for row in rows if (station_id := normalize_station_id(row.get("RENT_ID")))
    }
    return_ids = {
        station_id for row in rows if (station_id := normalize_station_id(row.get("RTN_ID")))
    }
    matched_rental_ids = rental_ids.intersection(stations)
    matched_return_ids = return_ids.intersection(stations)
    candidate_rentals = sum(
        bool(
            station_id
            and stations.get(station_id, StationInfo("", "", "", 0)).borough == candidate_borough
        )
        for row in rows
        if (station_id := normalize_station_id(row.get("RENT_ID")))
    )
    candidate_returns = sum(
        bool(
            station_id
            and stations.get(station_id, StationInfo("", "", "", 0)).borough == candidate_borough
        )
        for row in rows
        if (station_id := normalize_station_id(row.get("RTN_ID")))
    )

    query = ""
    if pages:
        query = f"{pages[0].rent_date}/{pages[0].rent_hour:02d}"
    reported_count = pages[0].reported_count if pages else 0
    return RentalAudit(
        query=query,
        response_service_names=tuple(sorted({page.response_service_name for page in pages})),
        result_codes=tuple(sorted({page.result_code for page in pages})),
        reported_count=reported_count,
        received_rows=len(rows),
        observed_fields=observed_fields,
        field_types=field_types,
        missing_values=missing_values,
        invalid_duration_values=sum(not _is_nonnegative_number(row.get("USE_MIN")) for row in rows),
        invalid_distance_values=sum(not _is_nonnegative_number(row.get("USE_DST")) for row in rows),
        unique_rental_station_ids=len(rental_ids),
        unique_return_station_ids=len(return_ids),
        rental_station_mapping_rate=_rate(len(matched_rental_ids), len(rental_ids)),
        return_station_mapping_rate=_rate(len(matched_return_ids), len(return_ids)),
        candidate_borough_rentals=candidate_rentals,
        candidate_borough_returns=candidate_returns,
    )


def build_historical_audit(
    station: StationAudit,
    inventory: InventoryAudit,
    rental: RentalAudit,
) -> HistoricalAudit:
    station_ready = (
        station.rows > 0
        and station.duplicate_station_ids == 0
        and station.missing_station_ids == 0
        and station.missing_boroughs == 0
        and station.missing_nominal_racks == 0
        and station.boroughs == 25
    )
    inventory_ready = (
        inventory.total_rows > 0
        and inventory.station_mapping_rate >= 0.95
        and inventory.target_month in {month.month for month in inventory.months}
        and all(
            month.missing_values == 0
            and month.invalid_dates == 0
            and month.invalid_hours == 0
            and month.invalid_inventory_values == 0
            and month.negative_inventory_values == 0
            and month.duplicate_station_time_keys == 0
            and len(month.missing_time_groups) <= 1
            and month.station_time_coverage_rate >= 0.98
            for month in inventory.months
        )
    )
    rental_ready = (
        rental.received_rows == rental.reported_count
        and rental.received_rows > 0
        and set(RENTAL_REQUIRED_FIELDS).issubset(rental.observed_fields)
        and rental.result_codes == ("INFO-000",)
        and rental.rental_station_mapping_rate >= 0.90
        and rental.return_station_mapping_rate >= 0.90
    )
    analysis_ready = station_ready and inventory_ready and rental_ready

    warnings: list[str] = []
    if rental.response_service_names != ("tbCycleRentData",):
        warnings.append("요청 서비스명은 tbCycleRentData지만 실제 응답 루트는 rentData로 관측됐다.")
    if "SEX_CD" not in rental.observed_fields:
        warnings.append("공식 출력 명세의 SEX_CD가 실제 표본 응답에는 없었다.")
    rental_missing_total = sum(rental.missing_values.values())
    if rental_missing_total:
        warnings.append(
            f"대여이력 표본의 필수 필드에 결측 {rental_missing_total:,}건이 있어 "
            "정제 시 별도 제외 또는 상태 분류가 필요하다."
        )
    if inventory.inventory_over_nominal_racks_rows:
        warnings.append(
            "과거 대여 가능 수량이 공식 거치대 수를 초과한 행이 있어 "
            "거치대 수를 하드 용량으로 사용할 수 없다."
        )
    missing_inventory_times = [
        timestamp for month in inventory.months for timestamp in month.missing_time_groups
    ]
    if missing_inventory_times:
        warnings.append("시간별 재고 스냅샷이 누락된 시각: " + ", ".join(missing_inventory_times))
    warnings.append(
        "2025년 12월 말 대여소 메타데이터를 2025년 4분기 전체에 대조했으므로 "
        "분기 중 신설·폐쇄 시차가 있을 수 있다."
    )
    warnings.append(
        "BIKE_ID·BIRTH_YEAR·USR_CLS_CD는 재배치 분석에 불필요하므로 정제 데이터에서 제외한다."
    )

    simulator_ready = analysis_ready and inventory.inventory_over_nominal_racks_rows == 0
    status = "FAIL"
    if analysis_ready:
        status = "PASS" if not warnings else "PASS_WITH_WARNINGS"

    return HistoricalAudit(
        audited_at_utc=datetime.now(UTC).isoformat(),
        status=status,
        analysis_ready=analysis_ready,
        simulator_ready=simulator_ready,
        selected_period=inventory.target_month,
        candidate_borough=inventory.candidate_borough,
        candidate_peak_hours=inventory.candidate_peak_hours,
        station=station,
        inventory=inventory,
        rental=rental,
        warnings=tuple(warnings),
        source_urls={
            "station_metadata": STATION_SOURCE_URL,
            "hourly_inventory": INVENTORY_SOURCE_URL,
            "rental_history": RENTAL_SOURCE_URL,
        },
    )


def write_historical_reports(
    audit: HistoricalAudit,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_historical_markdown(audit), encoding="utf-8")


def render_historical_markdown(audit: HistoricalAudit) -> str:
    month_rows = "\n".join(
        "| "
        f"{month.month} | {month.rows:,} | {month.time_groups:,} | "
        f"{len(month.missing_time_groups)} | {month.station_time_coverage_rate:.2%} | "
        f"{month.unique_station_ids:,} | "
        f"{month.matched_station_ids:,} | {month.zero_inventory_rows:,} | "
        f"{month.inventory_over_nominal_racks_rows:,} |"
        for month in audit.inventory.months
    )
    borough_rows = "\n".join(
        "| "
        f"{index} | {row.borough} | {row.unique_station_ids:,} | "
        f"{row.matched_capacity_rate:.2%} | {row.zero_inventory_rate:.2%} | "
        f"{row.over_nominal_racks_rate:.2%} | {', '.join(map(str, row.peak_zero_hours))} |"
        for index, row in enumerate(audit.inventory.borough_ranking[:10], start=1)
    )
    missing_rows = "\n".join(
        f"| `{field}` | {count:,} |" for field, count in audit.rental.missing_values.items()
    )
    warning_rows = "\n".join(f"- {warning}" for warning in audit.warnings)
    fields = ", ".join(f"`{field}`" for field in audit.rental.observed_fields)
    peak_hours = ", ".join(f"{hour:02d}시" for hour in audit.candidate_peak_hours)
    station_source = audit.source_urls["station_metadata"]
    inventory_source = audit.source_urls["hourly_inventory"]
    rental_source = audit.source_urls["rental_history"]
    station_type_counts = (
        f"{audit.station.lcd_station_rows:,} / {audit.station.qr_station_rows:,} / "
        f"{audit.station.hybrid_station_rows:,}"
    )
    nominal_rack_summary = (
        f"{audit.station.nominal_racks_min} / {audit.station.nominal_racks_median:g} / "
        f"{audit.station.nominal_racks_max}"
    )
    rental_roots = ", ".join(audit.rental.response_service_names)
    candidate_flow = (
        f"{audit.rental.candidate_borough_rentals:,} / {audit.rental.candidate_borough_returns:,}"
    )
    inventory_table_header = (
        "| 월 | 행 | 시간 그룹 | 누락 시간 | 대여소-시간 커버리지 | 고유 대여소 | "
        "메타데이터 매핑 | 0대 행 | 명목 거치대 초과 행 |"
    )

    return f"""# 과거 데이터 가용성·스키마 감사

- 감사시각(UTC): {audit.audited_at_utc}
- 결과: **{audit.status}**
- 분석 준비: **{"통과" if audit.analysis_ready else "실패"}**
- 양방향 시뮬레이터 준비: **{"통과" if audit.simulator_ready else "보류"}**
- 1차 분석기간: **{audit.selected_period}**
- 1차 후보 자치구: **{audit.candidate_borough}**
- 후보 집중 시간대: **{peak_hours}**

## 공식 출처와 사용 파일

| 데이터 | 공식 페이지 | 로컬 원천 파일 |
|---|---|---|
| 대여소 메타데이터 | {station_source} | `{audit.station.source_file}` |
| 시간별 대여 가능 수량 | {inventory_source} | `{audit.inventory.source_file}` |
| 대여이력 | {rental_source} | Open API `{audit.rental.query}` 표본 |

원천 파일은 `data/raw`에만 두며 Git에 포함하지 않는다.

## 대여소 메타데이터

- 행 / 고유 ID: {audit.station.rows:,} / {audit.station.unique_station_ids:,}
- 자치구: {audit.station.boroughs}
- ID 중복 / 결측: {audit.station.duplicate_station_ids:,} / {audit.station.missing_station_ids:,}
- 용량 결측: {audit.station.missing_nominal_racks:,}
- LCD / QR / 혼합 대여소 행: {station_type_counts}
- 명목 거치대 수 최소 / 중앙 / 최대: {nominal_rack_summary}

명목 거치대 수는 XLSX의 LCD 거치대 수와 QR 거치대 수를 합산했다.

## 시간별 재고 스냅샷

{inventory_table_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{month_rows}

- 전체 행: {audit.inventory.total_rows:,}
- 고유 대여소 매핑률: {audit.inventory.station_mapping_rate:.2%}
- 인코딩: `{audit.inventory.encoding}`
- 필드: {", ".join(f"`{field}`" for field in INVENTORY_FIELDS)}

`거치대수량`이라는 열 이름과 달리 값의 의미는 해당 시각의 **대여 가능 자전거 수**다.

## {audit.selected_period} 자치구 후보 순위

최소 75개 대여소와 95% 이상 용량 매핑을 만족하는 자치구 중 0대 관측률이 높은 곳을
1차 후보로 정했다. 아래 표는 비교를 위해 전체 자치구 상위 10개를 보여준다. 이 순위는
문제 발견용이며 운영 성과 순위가 아니다.

| 순위 | 자치구 | 대여소 | 용량 매핑률 | 0대 관측률 | 명목 거치대 초과율 | 집중 시간대 |
|---:|---|---:|---:|---:|---:|---|
{borough_rows}

## 대여이력 API 표본

- 요청 시간: `{audit.rental.query}`
- 문서 서비스명 / 실제 응답 루트: `tbCycleRentData` / `{rental_roots}`
- API 보고 건수 / 수신 건수: {audit.rental.reported_count:,} / {audit.rental.received_rows:,}
- 대여 대여소 ID 매핑률: {audit.rental.rental_station_mapping_rate:.2%}
- 반납 대여소 ID 매핑률: {audit.rental.return_station_mapping_rate:.2%}
- 후보 자치구 대여 / 반납: {candidate_flow}
- 관측 필드: {fields}

| 필수 필드 | 결측 건수 |
|---|---:|
{missing_rows}

## 경고와 결정

{warning_rows}

과거 데이터는 자치구별 재고 부족 분석과 수요 추정에 사용할 수 있다. 다만 공식 거치대 수가
물리적 하드 용량이 아니므로, 반납 실패까지 포함하는 양방향 시뮬레이터는 아직 착수하지 않는다.
다음 단계에서는 1차 후보 자치구의 한 달 대여·반납 흐름을 집계하고, 부족 감소만 평가하는
시뮬레이션 계약과 별도 유효용량 추정안을 비교한다.
"""


def normalize_station_id(value: object) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    if text.startswith("ST-"):
        text = text[3:]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


def _optional_nonnegative_int(value: object) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number >= 0 else None


def _is_nonnegative_number(value: object) -> bool:
    try:
        return float(_clean(value)) >= 0
    except ValueError:
        return False


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
