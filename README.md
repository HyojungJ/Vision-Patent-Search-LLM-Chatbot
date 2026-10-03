# 🔎 컴퓨터 비전 특허 검색·질의응답 RAG 챗봇

자연어로 질문하면 **유사 특허와 IPC 분류 코드를 찾아 근거를 들어 답하는** RAG 챗봇입니다.
먼저 텍스트 기반 특허 검색 엔진을 만들고 Django 웹 서비스로 확장했으며,
이후 **특허 도면(그림)까지 이해하는 Vision 모듈**을 추가해 멀티모달 방향으로 발전시키고 있습니다.

> **프로젝트 성격**
> SKN 부트캠프 팀 프로젝트(5인)로 특허 검색 엔진(1차) → Django 웹 서비스(2차)를 개발했고,
> 이후 개인적으로 구조를 리팩토링하고 **머신비전 방향의 업그레이드(Vision 모듈)** 를 이어서 진행 중입니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| 한 줄 소개 | 자연어 질문으로 특허·IPC를 검색하고 근거 기반으로 답하는 챗봇 + 특허 도면 번호 자동 인식 |
| 대상 사용자 | 개인 발명가, 중소기업, 특허 실무자의 초기 선행기술 조사 보조 |
| 핵심 접근 | RAG + LLM Agent의 Tool 자동 선택·연쇄 호출 |
| 데이터 | IPC 코드 약 7만 건, 공개 특허 약 3만 건, 청구항 약 59만 건 |
| 검색 방식 | ChromaDB 벡터 검색 + BM25 하이브리드 + 멀티 쿼리 Z-score 정규화 |
| 웹 서비스 | Django 회원·비회원 채팅, 실시간 스트리밍 |
| Vision(신규) | 특허 도면에서 부품 번호(101, 102…) 자동 인식(OCR) + 전처리 효과 검증 |
| 사용 기술 | Python, LangChain, LangGraph, ChromaDB, OpenAI API, Django, OpenCV, EasyOCR, Docker |

---

## 1. 이 프로젝트가 하는 일

크게 **두 축**으로 특허를 다룹니다.

- **특허 청구항 검색**: 기술 설명 → 의미가 비슷한 청구항 검색 → 특허 단위로 다시 정렬
- **IPC 코드 검색**: 기술 키워드 → IPC 후보 추천, 또는 입력한 IPC 코드의 뜻·계층 설명

여기에 더해, **특허 도면(그림)에서 부품 번호를 자동으로 읽는 기능(Vision 모듈)** 을 새로 붙이고 있습니다.
도면 속 번호와 설명 글("101: 하우징")을 연결하면, 글뿐 아니라 그림까지 이해하는 검색으로 확장할 수 있기 때문입니다.

---

## 2. 텍스트 RAG 챗봇 (기존)

### RAG Agent의 4개 Tool

| Tool | 역할 |
|------|------|
| `tool_search_patent_with_description` | 기술 설명으로 유사 특허·청구항 검색 |
| `tool_search_detail_patent_by_id` | 특정 출원번호의 특허 메타데이터·청구항 직접 조회 |
| `tool_search_ipc_code_with_description` | 기술 설명으로 IPC 후보 코드 추천 |
| `tool_search_ipc_description_from_code` | 입력된 IPC 코드의 의미와 계층 구조 조회 |

LLM Agent가 질문 유형(기술 설명 / IPC 코드 / 출원번호)에 따라 알맞은 Tool을 **스스로 골라 연쇄 호출**합니다.

### 검색 품질을 높인 방법

- **하이브리드 검색**: 벡터 유사도(0.7) + 키워드 기반 BM25(0.3)를 결합
- **멀티 쿼리 Z-score 정규화**: 여러 쿼리의 거리 분포 차이를 보정해 공정하게 비교
- **특허 단위 리랭킹**: 청구항 단위 결과를 출원번호로 묶어 특허 단위 점수로 재계산
- **평가 기준 설계**: "같은 IPC 코드를 공유하면 유사 특허"라는 도메인 기준으로 검색 품질을 수치로 검증

### Django 웹 서비스

- 토큰을 실시간으로 흘려 보여주는 **스트리밍 채팅 UI** (NDJSON)
- 회원은 사용자 ID, 비회원은 세션 ID로 대화방 분리 (게스트 → 회원가입 시 대화 자동 이관)
- 과거 대화를 LangChain 메시지로 변환해 **후속 질문의 문맥 유지**

---

## 3. 특허 도면 번호 읽기 (Vision 모듈, 신규)

> 쉽게 말해: **특허 그림 속 숫자(101, 102…)를 컴퓨터가 자동으로 읽게** 만든 기능입니다.

### 왜?
특허 도면에는 부품마다 번호가 있고, 설명 글에는 "101: 하우징"처럼 그 뜻이 적혀 있습니다.
번호를 자동으로 읽으면 **그림과 설명 글을 연결**할 수 있어서, 이것부터 해결했습니다.

### 무엇을 했나
1. **연습용 도면 만들기** — 진짜 도면은 아직 못 받아서(아래 참고), 특허 도면처럼 생긴 그림을 직접 생성. 깨끗한 버전 / 일부러 지저분하게(기울임·노이즈) 만든 버전 두 종류.
2. **그림 다듬기(전처리)** — 흑백 변환 → 노이즈 제거 → 기울어진 그림 똑바로 세우기. (3.5도 기울인 걸 3.49도로 거의 정확히 보정)
3. **번호 읽기(OCR)** — 글자 읽는 AI(EasyOCR)로 숫자 인식. **도면 속 번호 8개를 모두 정확히 읽음(100%).**

### 이 과정에서 알게 된 것
"그림을 다듬으면 더 잘 읽히겠지"라고 생각했지만, **실험해보니 반대**였습니다.
요즘 OCR AI가 워낙 똑똑해서, 어설픈 전처리는 오히려 숫자 획을 뭉개 방해가 됐습니다.
노이즈를 5단계로 올려가며 비교했고 **모든 단계에서 원본이 더 좋았습니다.**

| 노이즈 정도 | 전처리 안 함 | 전처리 함 |
|---|---|---|
| 깨끗함 | 100% | 87% |
| 약간 | 100% | 37% |
| 보통 | 100% | 37% |
| 심함 | 100% | 12% |
| 아주 심함 | 100% | 0% |

(숫자 = 도면 속 번호를 맞게 읽은 비율)

> 그래서 전처리를 "그냥 넣는" 대신, **실험으로 효과를 확인하고 이 경우엔 빼기로** 결정했습니다.

### 데이터에 대해
진짜 특허 도면은 **KIPRIS(특허 공공 데이터) API 승인**을 기다리는 중입니다.
번호 읽기 기능은 연습용 도면으로 이미 "된다"를 확인했고, 승인되면 **같은 코드로 실제 도면에 바로 적용**됩니다.
API로 도면을 받아오는 코드(`probe_kipris_api.py` 등)도 미리 만들어 두었습니다.

---

## 🗂️ 저장소 구조

```
Vision-Patent-Search-LLM-Chatbot
│
├─ rag_engine/                 # 특허·IPC RAG 검색 엔진
│   ├─ app/                    #   LangGraph ReAct Agent + Tool
│   │   ├─ main.py             #     에이전트 실행 및 대화 메모리
│   │   ├─ total_tools.py      #     특허·IPC 검색 Tool 4종
│   │   ├─ total_schemas.py    #     Pydantic 입출력 스키마
│   │   ├─ doc_func.py         #     벡터+BM25 특허 하이브리드 검색
│   │   └─ ipc_func.py         #     IPC 계층 검색·중복 제거
│   ├─ vision/                 #   ✨ 특허 도면 번호 읽기 (신규)
│   │   ├─ make_test_drawing.py       # 연습용 도면 생성
│   │   ├─ preprocess.py              # 그림 다듬기 (흑백·노이즈·기울기 보정)
│   │   ├─ reference_ocr.py           # 도면에서 부품 번호 읽기 (OCR)
│   │   ├─ benchmark_ocr.py           # "전처리가 도움 되나?" 비교 실험
│   │   ├─ debug_ocr.py               # 번호를 어디서 읽었는지 시각화
│   │   ├─ probe_kipris_api.py        # (승인 후) 특허 API에서 도면 받아오기
│   │   ├─ diagnose_kipris.py         # 특허 API 키·연결 상태 점검
│   │   └─ verify_drawing_download.py # 도면 이미지 다운로드 확인
│   ├─ db_search/              #   ChromaDB 구축·검색 최적화 노트북
│   ├─ doc/                    #   특허 청구항 전처리·임베딩·검색 실험
│   └─ ipc/                    #   IPC 코드 전처리·계층화·임베딩
│
└─ web_service/                # Django 웹 챗봇 서비스
    ├─ config/                 #   Django 설정·URL·WSGI/ASGI
    ├─ account/                #   회원가입·로그인·마이페이지
    ├─ chat/                   #   스트리밍 채팅·대화 내역 API
    ├─ main/                   #   메인 페이지
    └─ llm_module/             #   RAG 엔진을 웹 서비스용으로 이식
```

---

## 🛠️ 기술 스택

- **언어**: Python, HTML, CSS, JavaScript
- **LLM / Agent**: OpenAI API, LangChain, LangGraph
- **검색 / RAG**: ChromaDB, SentenceTransformers, BM25, 멀티 쿼리 검색
- **Vision**: OpenCV, EasyOCR
- **웹**: Django, Streamlit
- **데이터 검증 / 배포**: Pydantic, Docker, Nginx, Gunicorn

---

## 4. 앞으로 할 것 (Vision 다음 단계)

- 그림 속 번호 ↔ 설명 글 연결 (101 = 하우징)
- 비슷한 도면 찾기 (이미지 유사도 검색) — 실제 도면이 쌓여야 의미가 있어 데이터 수집 후 진행
- 챗봇에 도면 기능 붙이기 (멀티모달 검색 완성)

---

## 🌱 브랜치 안내

Vision 모듈 추가 전·후를 브랜치로 나눠 두어, 업그레이드 과정을 비교할 수 있습니다.

- `v1` : Vision 추가 **전** (텍스트 특허 검색 챗봇)
- `v2` / `main` : Vision 추가 **후** (이 문서 기준)

---

## 🚀 실행 방법

### 1) 환경 변수 준비
```powershell
Copy-Item .env.example rag_engine/.env
Copy-Item .env.example web_service/.env
```
생성한 `.env`에 `OPENAI_API_KEY`, `DJANGO_SECRET_KEY` 등 필요한 값을 입력합니다.
(Vision에서 실제 도면 수집 시에는 `KIPRIS_ACCESS_KEY`도 필요합니다.)

### 2) RAG 검색 엔진
```powershell
Set-Location rag_engine
pip install -r requirements.txt
streamlit run streamlit/app.py
```

### 3) Vision 모듈 (도면 번호 읽기 데모)
```powershell
Set-Location rag_engine/vision
pip install opencv-python easyocr
python make_test_drawing.py   # 연습용 도면 생성
python reference_ocr.py       # 번호 읽기 실행
python benchmark_ocr.py       # 전처리 효과 비교 실험
```

### 4) Django 웹 서비스
```powershell
Set-Location web_service
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

> 벡터 DB(`doc_db/`, `ipc_db/`)는 약 11GB의 대용량 실행 데이터라 저장소에 포함하지 않았습니다.
> `rag_engine`의 DB 구축 노트북으로 생성한 뒤 배치해야 검색이 동작합니다.
