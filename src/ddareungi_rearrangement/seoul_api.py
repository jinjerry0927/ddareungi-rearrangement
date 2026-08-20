from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

API_KEY_ENV_VAR = "SEOUL_OPEN_DATA_API_KEY"
DEFAULT_BASE_URL = "http://openapi.seoul.go.kr:8088"
REQUEST_SERVICE_NAME = "bikeList"
RESPONSE_SERVICE_NAMES = ("rentBikeStatus", "bikeList")
RENTAL_REQUEST_SERVICE_NAME = "tbCycleRentData"
RENTAL_RESPONSE_SERVICE_NAMES = ("rentData", "tbCycleRentData")


class SeoulOpenDataError(RuntimeError):
    """인증키를 노출하지 않는 서울 열린데이터광장 API 오류."""


@dataclass(frozen=True)
class LiveBikePage:
    start: int
    end: int
    response_service_name: str
    result_code: str
    reported_count: int
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RentalHistoryPage:
    start: int
    end: int
    rent_date: str
    rent_hour: int
    response_service_name: str
    result_code: str
    reported_count: int
    rows: tuple[dict[str, Any], ...]


class SeoulOpenDataClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise SeoulOpenDataError(f"{API_KEY_ENV_VAR} is not configured")

        self._api_key = clean_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> SeoulOpenDataClient:
        env_path = project_root / ".env" if project_root else None
        load_dotenv(env_path)
        return cls(os.getenv(API_KEY_ENV_VAR, ""))

    def __enter__(self) -> SeoulOpenDataClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_live_bike_page(self, start: int, end: int) -> LiveBikePage:
        if start < 1 or end < start:
            raise ValueError("start and end must define a positive inclusive range")
        if end - start + 1 > 1_000:
            raise ValueError("Seoul Open Data allows at most 1,000 rows per request")

        encoded_key = quote(self._api_key, safe="")
        url = f"{self._base_url}/{encoded_key}/json/{REQUEST_SERVICE_NAME}/{start}/{end}/"

        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise SeoulOpenDataError(
                f"Seoul API request failed ({type(exc).__name__}); credentials redacted"
            ) from None

        if response.status_code != 200:
            raise SeoulOpenDataError(
                f"Seoul API returned HTTP {response.status_code}; credentials redacted"
            )

        try:
            payload = response.json()
        except ValueError:
            raise SeoulOpenDataError(
                "Seoul API returned invalid JSON; credentials redacted"
            ) from None

        service_name, service = self._find_service(payload)
        result = service.get("RESULT", {})
        result_code = str(result.get("CODE", ""))
        if result_code != "INFO-000":
            raise SeoulOpenDataError(
                f"Seoul API returned result code {result_code or 'UNKNOWN'}; credentials redacted"
            )

        raw_rows = service.get("row", [])
        if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
            raise SeoulOpenDataError("Seoul API row schema is invalid; credentials redacted")

        try:
            reported_count = int(service.get("list_total_count", len(raw_rows)))
        except (TypeError, ValueError):
            raise SeoulOpenDataError(
                "Seoul API list_total_count is invalid; credentials redacted"
            ) from None

        return LiveBikePage(
            start=start,
            end=end,
            response_service_name=service_name,
            result_code=result_code,
            reported_count=reported_count,
            rows=tuple(dict(row) for row in raw_rows),
        )

    def iter_all_live_bike_pages(
        self,
        *,
        page_size: int = 1_000,
        max_pages: int = 10,
    ) -> Iterator[LiveBikePage]:
        if not 1 <= page_size <= 1_000:
            raise ValueError("page_size must be between 1 and 1,000")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")

        for page_index in range(max_pages):
            start = page_index * page_size + 1
            end = start + page_size - 1
            page = self.fetch_live_bike_page(start, end)
            yield page
            if len(page.rows) < page_size:
                return

        raise SeoulOpenDataError(
            "Live bike pagination reached max_pages before a final partial page"
        )

    def fetch_rental_history_page(
        self,
        start: int,
        end: int,
        *,
        rent_date: str,
        rent_hour: int,
    ) -> RentalHistoryPage:
        if start < 1 or end < start:
            raise ValueError("start and end must define a positive inclusive range")
        if end - start + 1 > 1_000:
            raise ValueError("Seoul Open Data allows at most 1,000 rows per request")
        try:
            date.fromisoformat(rent_date)
        except ValueError:
            raise ValueError("rent_date must use YYYY-MM-DD") from None
        if not 0 <= rent_hour <= 23:
            raise ValueError("rent_hour must be between 0 and 23")

        encoded_key = quote(self._api_key, safe="")
        url = (
            f"{self._base_url}/{encoded_key}/json/{RENTAL_REQUEST_SERVICE_NAME}/"
            f"{start}/{end}/{rent_date}/{rent_hour}"
        )
        payload = self._get_json(url)
        service_name, service = self._find_named_service(
            payload,
            RENTAL_RESPONSE_SERVICE_NAMES,
        )
        result_code, reported_count, rows = self._parse_service(service)

        return RentalHistoryPage(
            start=start,
            end=end,
            rent_date=rent_date,
            rent_hour=rent_hour,
            response_service_name=service_name,
            result_code=result_code,
            reported_count=reported_count,
            rows=rows,
        )

    def iter_all_rental_history_pages(
        self,
        *,
        rent_date: str,
        rent_hour: int,
        page_size: int = 1_000,
        max_pages: int = 100,
    ) -> Iterator[RentalHistoryPage]:
        if not 1 <= page_size <= 1_000:
            raise ValueError("page_size must be between 1 and 1,000")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")

        expected_total: int | None = None
        for page_index in range(max_pages):
            start = page_index * page_size + 1
            end = start + page_size - 1
            page = self.fetch_rental_history_page(
                start,
                end,
                rent_date=rent_date,
                rent_hour=rent_hour,
            )
            if expected_total is None:
                expected_total = page.reported_count
            elif page.reported_count != expected_total:
                raise SeoulOpenDataError(
                    "Rental history total count changed during pagination; credentials redacted"
                )

            yield page
            if end >= expected_total or len(page.rows) < page_size:
                return

        raise SeoulOpenDataError("Rental history pagination reached max_pages before completion")

    def _get_json(self, url: str) -> Any:
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise SeoulOpenDataError(
                f"Seoul API request failed ({type(exc).__name__}); credentials redacted"
            ) from None

        if response.status_code != 200:
            raise SeoulOpenDataError(
                f"Seoul API returned HTTP {response.status_code}; credentials redacted"
            )

        try:
            return response.json()
        except ValueError:
            raise SeoulOpenDataError(
                "Seoul API returned invalid JSON; credentials redacted"
            ) from None

    @staticmethod
    def _find_named_service(
        payload: Any,
        service_names: tuple[str, ...],
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise SeoulOpenDataError("Seoul API top-level schema is invalid; credentials redacted")

        for service_name in service_names:
            service = payload.get(service_name)
            if isinstance(service, dict):
                return service_name, service

        top_result = payload.get("RESULT")
        if isinstance(top_result, dict):
            result_code = str(top_result.get("CODE", "UNKNOWN"))
            raise SeoulOpenDataError(
                f"Seoul API returned result code {result_code}; credentials redacted"
            )

        raise SeoulOpenDataError("Seoul API service root is missing; credentials redacted")

    @staticmethod
    def _parse_service(
        service: dict[str, Any],
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        result = service.get("RESULT", {})
        result_code = str(result.get("CODE", ""))
        if result_code != "INFO-000":
            raise SeoulOpenDataError(
                f"Seoul API returned result code {result_code or 'UNKNOWN'}; credentials redacted"
            )

        raw_rows = service.get("row", [])
        if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
            raise SeoulOpenDataError("Seoul API row schema is invalid; credentials redacted")

        try:
            reported_count = int(service.get("list_total_count", len(raw_rows)))
        except (TypeError, ValueError):
            raise SeoulOpenDataError(
                "Seoul API list_total_count is invalid; credentials redacted"
            ) from None

        return result_code, reported_count, tuple(dict(row) for row in raw_rows)

    @staticmethod
    def _find_service(payload: Any) -> tuple[str, dict[str, Any]]:
        return SeoulOpenDataClient._find_named_service(payload, RESPONSE_SERVICE_NAMES)
