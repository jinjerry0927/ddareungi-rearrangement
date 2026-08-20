from ddareungi_rearrangement.live_audit import audit_live_bike_pages
from ddareungi_rearrangement.seoul_api import LiveBikePage


def test_live_audit_passes_complete_unique_rows() -> None:
    rows = (
        {
            "rackTotCnt": "10",
            "stationName": "one",
            "parkingBikeTotCnt": "3",
            "shared": "30",
            "stationLatitude": "37.5",
            "stationLongitude": "127.0",
            "stationId": "ST-1",
        },
        {
            "rackTotCnt": "0",
            "stationName": "two",
            "parkingBikeTotCnt": "1",
            "shared": "0",
            "stationLatitude": "37.6",
            "stationLongitude": "127.1",
            "stationId": "ST-2",
        },
    )
    page = LiveBikePage(
        start=1,
        end=2,
        response_service_name="rentBikeStatus",
        result_code="INFO-000",
        reported_count=2,
        rows=rows,
    )

    audit = audit_live_bike_pages([page])

    assert audit.status == "PASS"
    assert audit.total_rows == 2
    assert audit.duplicate_station_ids == 0
    assert audit.zero_rack_count == 1
    assert audit.parking_over_rack_count == 1
