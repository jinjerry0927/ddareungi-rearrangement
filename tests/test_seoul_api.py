import json

import httpx
import pytest

from ddareungi_rearrangement.seoul_api import SeoulOpenDataClient, SeoulOpenDataError


def _row(station_id: str) -> dict[str, str]:
    return {
        "rackTotCnt": "10",
        "stationName": f"station-{station_id}",
        "parkingBikeTotCnt": "3",
        "shared": "30",
        "stationLatitude": "37.5",
        "stationLongitude": "127.0",
        "stationId": station_id,
    }


def test_client_parses_current_response_root() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/json/bikeList/1/2/" in str(request.url)
        return httpx.Response(
            200,
            json={
                "rentBikeStatus": {
                    "list_total_count": 2,
                    "RESULT": {"CODE": "INFO-000", "MESSAGE": "ok"},
                    "row": [_row("ST-1"), _row("ST-2")],
                }
            },
        )

    with SeoulOpenDataClient("test-secret", transport=httpx.MockTransport(handler)) as client:
        page = client.fetch_live_bike_page(1, 2)

    assert page.response_service_name == "rentBikeStatus"
    assert page.reported_count == 2
    assert [row["stationId"] for row in page.rows] == ["ST-1", "ST-2"]


def test_pagination_stops_on_partial_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        rows = [_row("ST-1"), _row("ST-2")] if "/1/2/" in path else [_row("ST-3")]
        return httpx.Response(
            200,
            json={
                "rentBikeStatus": {
                    "list_total_count": len(rows),
                    "RESULT": {"CODE": "INFO-000", "MESSAGE": "ok"},
                    "row": rows,
                }
            },
        )

    with SeoulOpenDataClient("test-secret", transport=httpx.MockTransport(handler)) as client:
        pages = list(client.iter_all_live_bike_pages(page_size=2, max_pages=3))

    assert [len(page.rows) for page in pages] == [2, 1]


def test_errors_do_not_expose_api_key() -> None:
    secret = "never-expose-this-key"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"RESULT": {"CODE": "ERROR-300", "MESSAGE": "bad key"}}),
        )

    with SeoulOpenDataClient(secret, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SeoulOpenDataError) as error:
            client.fetch_live_bike_page(1, 1)

    assert secret not in str(error.value)


def test_client_parses_rental_history_response_drift() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/json/tbCycleRentData/1/2/2025-11-05/8" in str(request.url)
        return httpx.Response(
            200,
            json={
                "rentData": {
                    "list_total_count": "2",
                    "RESULT": {"CODE": "INFO-000", "MESSAGE": "ok"},
                    "row": [
                        {"RENT_ID": "00102", "RTN_ID": "00103"},
                        {"RENT_ID": "00103", "RTN_ID": "00102"},
                    ],
                }
            },
        )

    with SeoulOpenDataClient("test-secret", transport=httpx.MockTransport(handler)) as client:
        page = client.fetch_rental_history_page(
            1,
            2,
            rent_date="2025-11-05",
            rent_hour=8,
        )

    assert page.response_service_name == "rentData"
    assert page.reported_count == 2
    assert len(page.rows) == 2


def test_rental_history_pagination_uses_global_total() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        rows = (
            [{"RENT_ID": "00102"}, {"RENT_ID": "00103"}]
            if "/1/2/" in path
            else [{"RENT_ID": "00104"}]
        )
        return httpx.Response(
            200,
            json={
                "rentData": {
                    "list_total_count": "3",
                    "RESULT": {"CODE": "INFO-000", "MESSAGE": "ok"},
                    "row": rows,
                }
            },
        )

    with SeoulOpenDataClient("test-secret", transport=httpx.MockTransport(handler)) as client:
        pages = list(
            client.iter_all_rental_history_pages(
                rent_date="2025-11-05",
                rent_hour=8,
                page_size=2,
                max_pages=2,
            )
        )

    assert [len(page.rows) for page in pages] == [2, 1]
