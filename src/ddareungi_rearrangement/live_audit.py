from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ddareungi_rearrangement.seoul_api import LiveBikePage

REQUIRED_FIELDS = (
    "parkingBikeTotCnt",
    "rackTotCnt",
    "shared",
    "stationId",
    "stationLatitude",
    "stationLongitude",
    "stationName",
)
NUMERIC_FIELDS = (
    "parkingBikeTotCnt",
    "rackTotCnt",
    "shared",
    "stationLatitude",
    "stationLongitude",
)


@dataclass(frozen=True)
class PageAudit:
    requested_range: str
    response_service_name: str
    result_code: str
    reported_count: int
    row_count: int


@dataclass(frozen=True)
class LiveBikeAudit:
    audited_at_utc: str
    status: str
    total_rows: int
    unique_station_ids: int
    duplicate_station_ids: int
    observed_fields: tuple[str, ...]
    field_types: dict[str, tuple[str, ...]]
    missing_values: dict[str, int]
    invalid_numeric_values: dict[str, int]
    negative_inventory_values: int
    zero_rack_count: int
    parking_over_rack_count: int
    coordinates_outside_seoul_bounds: int
    pages: tuple[PageAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_live_bike_pages(pages: list[LiveBikePage]) -> LiveBikeAudit:
    rows = [row for page in pages for row in page.rows]
    observed_fields = tuple(sorted({field for row in rows for field in row}))
    station_ids = [str(row.get("stationId", "")).strip() for row in rows]

    field_types = {
        field: tuple(sorted({type(row.get(field)).__name__ for row in rows if field in row}))
        for field in observed_fields
    }
    missing_values = {
        field: sum(not str(row.get(field, "")).strip() for row in rows) for field in REQUIRED_FIELDS
    }
    invalid_numeric_values = {
        field: sum(not _is_decimal(row.get(field)) for row in rows) for field in NUMERIC_FIELDS
    }

    negative_inventory_values = sum(
        _as_decimal(row.get(field)) < 0
        for row in rows
        for field in ("parkingBikeTotCnt", "rackTotCnt")
        if _is_decimal(row.get(field))
    )
    zero_rack_count = sum(
        _as_decimal(row.get("rackTotCnt")) == 0
        for row in rows
        if _is_decimal(row.get("rackTotCnt"))
    )
    parking_over_rack_count = sum(
        _as_decimal(row.get("parkingBikeTotCnt")) > _as_decimal(row.get("rackTotCnt"))
        for row in rows
        if _is_decimal(row.get("parkingBikeTotCnt")) and _is_decimal(row.get("rackTotCnt"))
    )
    coordinates_outside_seoul_bounds = sum(
        not (
            Decimal("37.0") <= _as_decimal(row.get("stationLatitude")) <= Decimal("38.0")
            and Decimal("126.0") <= _as_decimal(row.get("stationLongitude")) <= Decimal("128.0")
        )
        for row in rows
        if _is_decimal(row.get("stationLatitude")) and _is_decimal(row.get("stationLongitude"))
    )

    duplicate_station_ids = len(station_ids) - len(set(station_ids))
    required_fields_present = set(REQUIRED_FIELDS).issubset(observed_fields)
    passed = (
        bool(rows)
        and required_fields_present
        and duplicate_station_ids == 0
        and all(count == 0 for count in missing_values.values())
        and all(count == 0 for count in invalid_numeric_values.values())
        and negative_inventory_values == 0
        and coordinates_outside_seoul_bounds == 0
    )

    page_audits = tuple(
        PageAudit(
            requested_range=f"{page.start}-{page.end}",
            response_service_name=page.response_service_name,
            result_code=page.result_code,
            reported_count=page.reported_count,
            row_count=len(page.rows),
        )
        for page in pages
    )

    return LiveBikeAudit(
        audited_at_utc=datetime.now(UTC).isoformat(),
        status="PASS" if passed else "FAIL",
        total_rows=len(rows),
        unique_station_ids=len(set(station_ids)),
        duplicate_station_ids=duplicate_station_ids,
        observed_fields=observed_fields,
        field_types=field_types,
        missing_values=missing_values,
        invalid_numeric_values=invalid_numeric_values,
        negative_inventory_values=negative_inventory_values,
        zero_rack_count=zero_rack_count,
        parking_over_rack_count=parking_over_rack_count,
        coordinates_outside_seoul_bounds=coordinates_outside_seoul_bounds,
        pages=page_audits,
    )


def write_audit_reports(
    audit: LiveBikeAudit,
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
    markdown_path.write_text(render_audit_markdown(audit), encoding="utf-8")


def render_audit_markdown(audit: LiveBikeAudit) -> str:
    page_rows = "\n".join(
        f"| {page.requested_range} | {page.response_service_name} | "
        f"{page.result_code} | {page.reported_count} | {page.row_count} |"
        for page in audit.pages
    )
    missing_rows = "\n".join(
        f"| `{field}` | {count} |" for field, count in audit.missing_values.items()
    )
    numeric_rows = "\n".join(
        f"| `{field}` | {count} |" for field, count in audit.invalid_numeric_values.items()
    )
    fields = ", ".join(f"`{field}`" for field in audit.observed_fields)

    return f"""# 실시간 따릉이 API 스키마 감사

- 감사시각(UTC): {audit.audited_at_utc}
- 결과: **{audit.status}**
- 전체 행: {audit.total_rows:,}
- 고유 대여소 ID: {audit.unique_station_ids:,}
- 중복 대여소 ID: {audit.duplicate_station_ids:,}

## 페이지 응답

| 요청 범위 | 실제 응답 루트 | 결과 코드 | API 보고 건수 | 수신 행 |
|---|---|---|---:|---:|
{page_rows}

요청 서비스명은 `bikeList`이지만 현재 실제 JSON 응답 루트는 `rentBikeStatus`다.
`list_total_count`는 전체 대여소 수가 아니라 각 페이지에서 반환한 행 수로 관측됐다.

## 관측 필드

{fields}

현재 7개 필드는 모두 문자열로 반환되므로 정규화 단계에서 수치형으로 변환해야 한다.

## 결측

| 필드 | 결측 건수 |
|---|---:|
{missing_rows}

## 수치 변환 오류

| 필드 | 변환 불가 건수 |
|---|---:|
{numeric_rows}

## 추가 진단

- 음수 재고·거치대 값: {audit.negative_inventory_values:,}
- 거치대 수가 0인 대여소: {audit.zero_rack_count:,}
- 대여 가능 자전거가 거치대 수보다 많은 대여소: {audit.parking_over_rack_count:,}
- 서울 범위 밖 좌표: {audit.coordinates_outside_seoul_bounds:,}

`parkingBikeTotCnt > rackTotCnt`는 공유 거치 방식 때문에 가능할 수 있으므로 오류로 판정하지 않고
진단값으로만 기록한다.

## 판정

이 감사의 PASS는 API 연결, 페이지 수집, 필수 필드 존재, 결측·중복·수치 변환 가능 여부가
M0 실시간 스냅샷 수집을 시작하기에 충분하다는 뜻이다. 과거 재고 스냅샷과 대여이력 데이터의
가용성을 통과했다는 뜻은 아니며, 시뮬레이터 착수 승인은 별도 데이터 감사 후 결정한다.
"""


def _is_decimal(value: object) -> bool:
    try:
        Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return False
    return True


def _as_decimal(value: object) -> Decimal:
    return Decimal(str(value).strip())
