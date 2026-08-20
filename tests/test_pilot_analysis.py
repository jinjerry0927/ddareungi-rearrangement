import csv
import io
import zipfile

import polars as pl

from ddareungi_rearrangement.historical_audit import StationInfo
from ddareungi_rearrangement.pilot_analysis import (
    INVENTORY_FIELDS,
    RENTAL_FIELDS,
    extract_borough_inventory,
    extract_borough_trips,
)


def _stations() -> dict[str, StationInfo]:
    return {
        "102": StationInfo("102", "강남역", "강남구", 10),
        "103": StationInfo("103", "역삼역", "강남구", 12),
        "201": StationInfo("201", "잠실역", "송파구", 15),
    }


def _rental_row(
    rent_id: str,
    return_id: str,
    *,
    return_at: str = "2025-11-01 00:20:00",
) -> list[str]:
    return [
        "SPB-1",
        "2025-11-01 00:00:00",
        rent_id,
        "대여소",
        "0",
        return_at,
        return_id,
        "반납소" if return_id else "",
        "0",
        "20",
        "1000.5",
        "1990",
        "M",
        "내국인",
        f"ST-{int(rent_id)}",
        f"ST-{int(return_id)}" if return_id else "",
        "일반자전거",
    ]


def _write_csv_zip(path: object, name: str, header: tuple[str, ...], rows: list[list[str]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, buffer.getvalue().encode("cp949"))


def test_trip_extraction_keeps_boundary_flows_and_drops_personal_fields(tmp_path) -> None:
    source = tmp_path / "rentals.zip"
    output = tmp_path / "trips.parquet"
    _write_csv_zip(
        source,
        "rentals_2511.csv",
        RENTAL_FIELDS,
        [
            _rental_row("00102", "00103"),
            _rental_row("00102", "00201"),
            _rental_row("00201", "00103"),
            _rental_row("00201", "00201"),
            _rental_row("00102", "", return_at=""),
        ],
    )

    audit = extract_borough_trips(
        source,
        _stations(),
        month="2025-11",
        borough="강남구",
        output_path=output,
        batch_size=2,
    )
    frame = pl.read_parquet(output)

    assert audit.source_rows == 5
    assert audit.scoped_rows == 4
    assert audit.flow_counts == {
        "inbound": 1,
        "internal": 1,
        "outbound": 1,
        "unresolved_return": 1,
    }
    assert frame.height == 4
    assert "자전거번호" not in frame.columns
    assert "생년" not in frame.columns
    assert "성별" not in frame.columns


def test_inventory_extraction_filters_to_scope(tmp_path) -> None:
    source = tmp_path / "inventory.zip"
    output = tmp_path / "inventory.parquet"
    _write_csv_zip(
        source,
        "inventory_2511.csv",
        INVENTORY_FIELDS,
        [
            ["2025-11-01", "00102", "강남역", "0", "0"],
            ["2025-11-01", "00103", "역삼역", "0", "13"],
            ["2025-11-01", "00201", "잠실역", "0", "2"],
        ],
    )

    audit = extract_borough_inventory(
        source,
        _stations(),
        month="2025-11",
        borough="강남구",
        output_path=output,
        batch_size=1,
    )
    frame = pl.read_parquet(output)

    assert audit.source_rows == 3
    assert audit.scoped_rows == 2
    assert audit.unique_station_ids == 2
    assert frame["is_empty"].to_list() == [True, False]
    assert frame["over_nominal"].to_list() == [False, True]
