# 따릉이 재배치 정책 시뮬레이터

서울시 공공자전거 따릉이의 품절을 줄이면서도, 공급 대여소의 이용자를 과도하게 희생하지
않는 재배치 정책을 찾는 데이터 분석 프로젝트입니다. 2025년 11월 강남구 대여·재고 이력을
이벤트 단위로 재생하고, 정책별 서비스·형평성·운영량의 trade-off를 검증합니다.

현재 단계는 **M9: donor reserve 7 단일 홀드아웃 검증 완료**입니다.

![P0·P2·P3 donor reserve 홀드아웃 비교](reports/figures/gangnam_2025_11_donor_reserve_holdout.png)

## 핵심 질문과 결론

재배치가 필요한 수요처만 보면 공급 대여소에서 자전거를 많이 가져올수록 유리합니다. 하지만
그 결과 원래 성공했을 공급지 이용자가 실패할 수 있습니다. 그래서 공급지에 반드시 남길 최소
재고인 `donor reserve`를 정책 변수로 두었습니다.

학습기간 Pareto 분석에서 서비스 우선 `reserve 5`와 요청 보호를 강화한 `reserve 7` 사이의
절충을 확인한 뒤, **결과를 보기 전에 reserve 7을 동결**하고 11월 24~28일 홀드아웃을 한 번만
실행했습니다.

| 정책 | 성공률 | 실패 | P0 성공→정책 실패 | 대여소 p10 | P0보다 악화된 대여소 | 이동 대수 | 거리 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 재배치 없음 | 88.24% | 1,905건 | 0건 | 54.07% | 0곳 | 0대 | 0.0km |
| P2 서비스형, reserve 5 | 95.92% | 661건 | 122건 | 89.09% | 17곳 | 1,435대 | 336.8km |
| **P3 보호형, reserve 7** | **95.74%** | **691건** | **100건** | **87.80%** | **12곳** | **1,224대** | **359.2km** |

P3는 P2보다 전체 실패가 30건 늘고 p10이 1.286%p 낮아졌습니다. 반면 P0에서는 성공했지만
정책 때문에 실패한 요청을 22건 줄였고, 악화 대여소도 17곳에서 12곳으로 줄였습니다. 이동
대수는 211대 감소했으나 더 먼 공급지를 이용해 총 거리는 22.4km 늘었습니다.

따라서 이 프로젝트의 결론은 “reserve 7이 무조건 최적”이 아닙니다. **전체 서비스 30건과
개별 이용자 악화 22건 사이에서 공급지 보호를 택한 운영안**이며, 데이터만으로 없앨 수 없는
가치 판단을 수치로 드러낸 결과입니다.

상세 결과는 [단일 홀드아웃 보고서](reports/gangnam_2025_11_donor_reserve_holdout.md)에서
확인할 수 있습니다.

## 데이터와 실험 범위

- 서울 전체 2025년 11월 대여이력 3,186,968행을 스트리밍 처리
- 강남구 출발 또는 도착 이동 137,463건만 정제
- 자전거번호·생년·성별·이용자종류는 저장하지 않음
- 분석 후보 167개 대여소 중 좌표가 확인된 165개를 모든 공간 정책에 공통 적용
- 학습기간: 2025-11-03 00:00 이상, 2025-11-22 00:00 미만
- 홀드아웃: 2025-11-24 00:00 이상, 2025-11-29 00:00 미만
- 최종 홀드아웃 관측 요청: 16,205건

```mermaid
flowchart LR
    A["서울시 원천 데이터"] --> B["스키마·커버리지 감사"]
    B --> C["강남구 비식별 정제"]
    C --> D["이벤트 기반 관측수요 재생"]
    D --> E["학습기간 정책·Pareto 비교"]
    E --> F["정책·지표 동결"]
    F --> G["단일 홀드아웃"]
    G --> H["서비스·피해·형평성·운영량 보고"]
```

## 시뮬레이터가 반영하는 것

- 동일 초기 재고와 동일 요청 순서에서 P0·P1·P2·P3를 결정론적으로 비교
- 대여 성공/실패, 내부 반납, 강남구 외부 유입·유출 이벤트를 시간순으로 처리
- 실패한 대여의 후속 반납을 억제해 존재하지 않는 자전거가 생기지 않도록 처리
- Haversine 거리, 도로거리 보정, 평균속도, 상하차 시간에 따른 재배치 지연 도착
- 시간당 작업 수, 차량 적재량, 이동 대수 상한 적용
- 요청별 구제·악화 trace와 대여소별 개선·악화 분포 산출
- 사용자 이동 중·재배치 중 자전거까지 포함한 보존식 검증

주요 구현은 [simulation.py](src/ddareungi_rearrangement/simulation.py), 실행 인터페이스는
[cli.py](src/ddareungi_rearrangement/cli.py), 회귀 테스트는
[test_simulation.py](tests/test_simulation.py)에서 확인할 수 있습니다.

## 분석 단계별 결과

| 분석 | 확인한 내용 | 보고서 |
|---|---|---|
| 데이터 감사 | 재고·대여이력·대여소 메타데이터의 분석 가능 범위와 경고 | [과거 데이터 감사](reports/historical_data_audit.md) |
| 강남구 기준선 | 품절 시간대, 대여소 순유입, 대여 흐름 | [기준선 EDA](reports/gangnam_2025_11_baseline.md) |
| P0/P1 재생 | 고정 임계값 정책의 낙관적 상한 | [기본 시뮬레이션](reports/gangnam_2025_11_simulation.md) |
| P0/P1/P2 | 거리·시간 지연이 있는 greedy-nearest 비교 | [공간 시뮬레이션](reports/gangnam_2025_11_spatial_simulation.md) |
| 운영 민감도 | 작업 1~3회, 속도 10~20km/h, 적재 10·20대 18개 조합 | [민감도 분석](reports/gangnam_2025_11_p2_sensitivity.md) |
| 시간 강건성 | 30일 모두 P2가 P0보다 개선되는지 날짜별 확인 | [일별 강건성](reports/gangnam_2025_11_daily_robustness.md) |
| 공간 형평성 | 개선·동률·악화 대여소와 p10 서비스율 | [대여소 형평성](reports/gangnam_2025_11_station_equity.md) |
| 요청 피해 | P0 성공에서 정책 실패로 바뀐 요청과 선행 유출 추적 | [요청 trace](reports/gangnam_2025_11_harm_trace.md) |
| P3 학습 | reserve 5·6·7·8의 비가중 Pareto 비교 | [reserve 학습](reports/gangnam_2025_11_donor_reserve_training.md) |
| P3 검증 | 동결한 reserve 7과 기존 reserve 5의 단일 홀드아웃 | [reserve 홀드아웃](reports/gangnam_2025_11_donor_reserve_holdout.md) |

## 재현 방법

### 1. 개발환경

- Python 3.11
- `uv` 패키지·가상환경 관리
- Polars, PyArrow, DuckDB, Matplotlib
- pytest, Ruff

```powershell
git clone https://github.com/jinjerry0927/ddareungi-rearrangement.git
Set-Location ddareungi-rearrangement
uv sync
uv run ddareungi doctor
```

### 2. 원천 데이터

대용량 원천·처리 데이터는 Git에 포함하지 않습니다. 공식 페이지에서 받은 파일을 다음 위치에
준비해야 전체 분석을 재현할 수 있습니다.

- `data/raw/stations_2025_12.xlsx`
- `data/raw/inventory_2025_q4.zip`
- `data/raw/rental_history_2025.zip`

실시간 API 감사와 좌표 스냅샷에는 서울 열린데이터광장 인증키가 필요합니다.

```powershell
Copy-Item -LiteralPath '.env.example' -Destination '.env'
```

```dotenv
SEOUL_OPEN_DATA_API_KEY=발급받은_인증키
```

`.env`는 Git에서 제외됩니다. 인증키를 커밋하거나 보고서에 기록하지 않습니다.

### 3. 분석 파이프라인

```powershell
uv run ddareungi audit-historical-data
uv run ddareungi build-pilot
uv run ddareungi run-simulation
uv run ddareungi snapshot-coordinates
uv run ddareungi run-spatial-simulation
uv run ddareungi run-sensitivity
uv run ddareungi run-temporal-robustness
uv run ddareungi run-station-equity
uv run ddareungi run-harm-trace
uv run ddareungi run-donor-reserve-training
uv run ddareungi run-donor-reserve-holdout
```

전체 코드 품질·회귀 테스트·환경 진단은 한 번에 실행할 수 있습니다.

```powershell
.\scripts\verify.ps1
```

현재 검증 기준은 Ruff lint·format, pytest **25개**, Python·필수 디렉터리·API 키 설정 확인
통과입니다. 분석 명령은 기본 경로 대신 각 입력·출력 경로를 CLI 옵션으로 바꿀 수 있습니다.

## 해석할 때 지켜야 할 한계

- 공개 대여이력에는 성공한 요청만 있어, 품절 때문에 시도하지 못한 잠재 수요는 빠져 있습니다.
- 공식 거치대 수보다 재고가 많은 관측이 빈번해 만차·반납 실패는 아직 모델링하지 않습니다.
- 좌표는 현재 API 스냅샷이라 2025년 운영 당시 위치와 다를 수 있습니다.
- 차량은 공급 대여소에서 바로 출발하며 첫 접근 이동·연속 경로·실제 교통은 미반영입니다.
- 일별 비교는 실제 자정 재고로 매일 초기화해 월 연속 무재배치와 계약이 다릅니다.
- 서비스 P2 운영능력과 형평성도 같은 5일 홀드아웃에서 분석돼 완전히 독립된 검증이 아닙니다.
- 대여소별 서비스율은 운영 형평성 대리변수이며 인구·소득·교통약자를 직접 나타내지 않습니다.

이 결과는 현장 효과의 확정치가 아니라, **정책 가정이 같은 관측수요 재생 안에서 어떤 요청과
대여소에 이익·피해를 옮기는지 비교하는 의사결정 실험**입니다.

## 프로젝트 문서

- [전체 프로젝트 계획](docs/PROJECT_PLAN.md)
- [현재 상태와 검증 기록](docs/STATE.md)
- [자율 실행 루프와 중단 규칙](LOOP.md)

## 데이터 출처

- [서울시 공공자전거 따릉이 대여이력 정보](https://data.seoul.go.kr/dataList/OA-15182/A/1/datasetView.do)
- [서울시 대여소별 공공자전거 따릉이 대여가능 수량](https://data.seoul.go.kr/dataList/OA-22382/F/1/datasetView.do)
- [서울시 공공자전거 따릉이 대여소 정보](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do)
- [서울시 공공자전거 따릉이 실시간 대여정보](https://data.seoul.go.kr/dataList/OA-15493/A/1/datasetView.do)
