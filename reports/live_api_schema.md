# 실시간 따릉이 API 스키마 감사

- 감사시각(UTC): 2026-08-19T07:08:48.595548+00:00
- 결과: **PASS**
- 전체 행: 2,736
- 고유 대여소 ID: 2,736
- 중복 대여소 ID: 0

## 페이지 응답

| 요청 범위 | 실제 응답 루트 | 결과 코드 | API 보고 건수 | 수신 행 |
|---|---|---|---:|---:|
| 1-1000 | rentBikeStatus | INFO-000 | 1000 | 1000 |
| 1001-2000 | rentBikeStatus | INFO-000 | 1000 | 1000 |
| 2001-3000 | rentBikeStatus | INFO-000 | 736 | 736 |

요청 서비스명은 `bikeList`이지만 현재 실제 JSON 응답 루트는 `rentBikeStatus`다.
`list_total_count`는 전체 대여소 수가 아니라 각 페이지에서 반환한 행 수로 관측됐다.

## 관측 필드

`parkingBikeTotCnt`, `rackTotCnt`, `shared`, `stationId`, `stationLatitude`, `stationLongitude`, `stationName`

현재 7개 필드는 모두 문자열로 반환되므로 정규화 단계에서 수치형으로 변환해야 한다.

## 결측

| 필드 | 결측 건수 |
|---|---:|
| `parkingBikeTotCnt` | 0 |
| `rackTotCnt` | 0 |
| `shared` | 0 |
| `stationId` | 0 |
| `stationLatitude` | 0 |
| `stationLongitude` | 0 |
| `stationName` | 0 |

## 수치 변환 오류

| 필드 | 변환 불가 건수 |
|---|---:|
| `parkingBikeTotCnt` | 0 |
| `rackTotCnt` | 0 |
| `shared` | 0 |
| `stationLatitude` | 0 |
| `stationLongitude` | 0 |

## 추가 진단

- 음수 재고·거치대 값: 0
- 거치대 수가 0인 대여소: 0
- 대여 가능 자전거가 거치대 수보다 많은 대여소: 1,023
- 서울 범위 밖 좌표: 0

`parkingBikeTotCnt > rackTotCnt`는 공유 거치 방식 때문에 가능할 수 있으므로 오류로 판정하지 않고
진단값으로만 기록한다.

## 판정

이 감사의 PASS는 API 연결, 페이지 수집, 필수 필드 존재, 결측·중복·수치 변환 가능 여부가
M0 실시간 스냅샷 수집을 시작하기에 충분하다는 뜻이다. 과거 재고 스냅샷과 대여이력 데이터의
가용성을 통과했다는 뜻은 아니며, 시뮬레이터 착수 승인은 별도 데이터 감사 후 결정한다.
