# ddareungi-rearrangement

서울시 공공자전거 따릉이의 대여·반납 실패를 줄이기 위해 재배치 정책을 데이터로 비교하는 프로젝트입니다.

현재 단계는 **M2: 강남구 관측수요 재생 시뮬레이터**입니다. 무재배치(P0)와 고정 임계값(P1)을 동일한 초기 재고·요청 순서로 비교합니다.

## 개발환경

- Python 3.11
- VS Code + Python/Jupyter 확장
- `uv` 패키지 및 가상환경 관리
- DuckDB, Polars, PyArrow, fastexcel 기반 데이터 처리
- pytest, Ruff 기반 검증

## 빠른 시작

```powershell
uv sync
uv run ddareungi doctor
uv run ddareungi audit-live-api
uv run ddareungi audit-historical-data
uv run ddareungi build-pilot
uv run ddareungi run-simulation
uv run pytest
uv run ruff check .
```

전체 검증은 다음 명령으로 실행합니다.

```powershell
.\scripts\verify.ps1
```

## API 키 설정

서울 열린데이터광장 인증키를 발급받은 뒤 `.env.example`을 `.env`로 복사하고 값을 입력합니다.

```powershell
Copy-Item -LiteralPath '.env.example' -Destination '.env'
```

```dotenv
SEOUL_OPEN_DATA_API_KEY=발급받은_인증키
```

`.env`는 Git에서 제외됩니다. 인증키를 커밋하거나 채팅에 붙여넣지 마세요.

## 디렉터리

- `data/raw`: 내려받은 원천 데이터, Git 제외
- `data/interim`: 정제 중간 결과, Git 제외
- `data/processed`: 분석용 결과, Git 제외
- `data/sample`: 테스트와 재현을 위한 소형 샘플
- `notebooks`: 데이터 감사와 설명용 Notebook
- `src/ddareungi_rearrangement`: 재사용 가능한 프로그램 코드
- `tests`: 단위·통합 테스트
- `reports`: 데이터 감사와 분석 보고서
- `docs`: 계획과 프로젝트 상태

## 현재 목표

첫 번째 데이터 목표는 한 달치 원천 데이터를 이용해 자치구별로 다음 품질표를 재현하는 것입니다.

- 활성 대여소 수
- 대여·반납 거래량
- 대여소 ID 매핑률
- 대여소 용량 확보율
- 재고 스냅샷 커버리지
- 분석 후보 자치구의 통과·실패 사유

현재 실시간 API 연결과 스키마 감사는 다음 명령으로 재현합니다. 인증키 값은 출력하거나
보고서에 저장하지 않습니다.

```powershell
uv run ddareungi audit-live-api
```

과거 데이터 감사는 공식 페이지에서 받은 다음 두 파일이 필요합니다. 원본은 Git에서 제외됩니다.

- `data/raw/stations_2025_12.xlsx`: 공공자전거 대여소 정보(25.12월 기준)
- `data/raw/inventory_2025_q4.zip`: 대여소별 대여 가능 수량(1시간 단위), 2025년 4분기
- `data/raw/rental_history_2025.zip`: 2025년 공공자전거 대여이력

```powershell
uv run ddareungi audit-historical-data
```

감사 결과는 `reports/historical_data_audit.md`에 기록됩니다. 현재 판정은 2025년 11월 강남구의
재고 부족 분석은 가능하지만, 공식 거치대 수를 물리적 하드 용량으로 쓸 수 없어 반납 실패까지
포함하는 양방향 시뮬레이터는 보류한다는 것입니다.

강남구 파일럿 기준선은 다음 명령으로 재현합니다.

```powershell
uv run ddareungi build-pilot
```

이 명령은 서울 전체 11월 대여이력 3,186,968행을 스트리밍하며 강남구를 출발하거나 도착하는
137,463건만 정제합니다. 자전거번호·생년·성별·이용자종류는 저장하지 않습니다. 결과 보고서는
`reports/gangnam_2025_11_baseline.md`, 차트는 `reports/figures`에 생성됩니다.

정책 비교 시뮬레이션은 다음 명령으로 재현합니다.

```powershell
uv run ddareungi run-simulation
```

11월 3~21일에서 세 가지 고정 임계값 후보를 비교하고, 선택된 정책을 11월 24~28일 홀드아웃에서
무재배치와 비교합니다. 공개 데이터에 기록된 성공 요청만 재생하고 반납 용량·차량 이동시간은
아직 모델링하지 않으므로 결과는 현장 효과가 아닌 낙관적 스트레스 테스트로 해석해야 합니다.
상세 결과는 `reports/gangnam_2025_11_simulation.md`에 생성됩니다.

세부 범위와 중단 규칙은 [프로젝트 계획](docs/PROJECT_PLAN.md)과 [현재 상태](docs/STATE.md)를 따릅니다.

## 데이터 출처

- [서울시 공공자전거 따릉이 대여이력 정보](https://data.seoul.go.kr/dataList/OA-15182/A/1/datasetView.do)
- [서울시 대여소별 공공자전거 따릉이 대여가능 수량](https://data.seoul.go.kr/dataList/OA-22382/F/1/datasetView.do)
- [서울시 공공자전거 따릉이 대여소 정보](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do)
- [서울시 공공자전거 따릉이 실시간 대여정보](https://data.seoul.go.kr/dataList/OA-15493/A/1/datasetView.do)
