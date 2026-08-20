import csv
import io
import zipfile

import polars as pl

from ddareungi_rearrangement.historical_audit import (
    audit_inventory_zip,
    audit_rental_pages,
    audit_station_frame,
    build_historical_audit,
    normalize_station_id,
)
from ddareungi_rearrangement.seoul_api import RentalHistoryPage


def _station_frame() -> pl.DataFrame:
    rows = [
        [
            "대여소\n번호",
            "보관소(대여소)명",
            "소재지(위치)",
            None,
            None,
            None,
            "설치\n시기",
            "설치형태",
            None,
            "운영\n방식",
        ],
        [None, None, None, None, None, None, None, "LCD", "QR", None],
        [None, None, "자치구", "상세주소", "위도", "경도", None, None, None, None],
        [None, None, None, None, None, None, None, "거치\n대수", "거치\n대수", None],
        [None] * 10,
        [102, "망원역", "마포구", "주소", 37.5, 126.9, "2020-01-01", None, 10, "QR"],
        [103, "합정역", "마포구", "주소", 37.5, 126.9, "2020-01-01", 8, None, "LCD"],
    ]
    return pl.DataFrame(rows, orient="row", schema=[f"column_{index}" for index in range(10)])


def _write_inventory_zip(path: object) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["일시", "대여소번호", "대여소명", "시간대", "거치대수량"])
    writer.writerow(["2025-11-01", "00102", "망원역", "8", "0"])
    writer.writerow(["2025-11-01", "00103", "합정역", "8", "9"])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data_2511.csv", buffer.getvalue().encode("cp949"))


def _rental_page() -> RentalHistoryPage:
    row = {
        "RENT_DT": "2025-11-05 08:00:00",
        "RENT_ID": "00102",
        "RENT_NM": "망원역",
        "RTN_DT": "2025-11-05 08:10:00",
        "RTN_ID": "00103",
        "RTN_NM": "합정역",
        "USE_MIN": "10",
        "USE_DST": "1200",
        "RENT_STATION_ID": "ST-102",
        "RETURN_STATION_ID": "ST-103",
    }
    return RentalHistoryPage(
        start=1,
        end=1,
        rent_date="2025-11-05",
        rent_hour=8,
        response_service_name="rentData",
        result_code="INFO-000",
        reported_count=1,
        rows=(row,),
    )


def test_historical_audit_detects_capacity_semantics_warning(tmp_path) -> None:
    station_dataset = audit_station_frame(_station_frame())
    inventory_path = tmp_path / "inventory.zip"
    _write_inventory_zip(inventory_path)

    inventory = audit_inventory_zip(
        inventory_path,
        station_dataset.stations,
        target_month="2025-11",
    )
    rental = audit_rental_pages(
        [_rental_page()],
        station_dataset.stations,
        candidate_borough=inventory.candidate_borough,
    )
    audit = build_historical_audit(station_dataset.audit, inventory, rental)

    assert inventory.inventory_over_nominal_racks_rows == 1
    assert inventory.zero_inventory_rows == 1
    assert audit.analysis_ready is False  # fixture has only one of Seoul's 25 boroughs
    assert audit.simulator_ready is False
    assert any("하드 용량" in warning for warning in audit.warnings)


def test_station_id_normalization_handles_all_observed_formats() -> None:
    assert normalize_station_id("00102") == "102"
    assert normalize_station_id("ST-102") == "102"
    assert normalize_station_id(102.0) == "102"
