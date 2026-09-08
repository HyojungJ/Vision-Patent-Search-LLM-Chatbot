# 🔎 Vision-Patent-Search-LLM-Chatbot

컴퓨터 비전 분야 **특허 검색·질의응답 LLM 챗봇**. RAG 검색 엔진을 만들고(3차), 이를 Django 웹 서비스로 배포한(4차) 팀 프로젝트를 개인적으로 정리·복기한 저장소입니다.

> **이 저장소에 대하여**
> SKN 부트캠프 팀 프로젝트(FantAstIc 5, 5인)로 진행했습니다.
> 두 개의 연속된 프로젝트 — **RAG 검색 엔진(3차)** 과 이를 웹으로 배포한 **Django 서비스(4차)** — 를 하나의 저장소에 정리했습니다.
> 팀 결과물을 기반으로, 전체 구조를 다시 이해하고 흩어져 있던 코드를 재배치하며, 제가 맡은 부분과 배운 점을 정리하는 데 초점을 맞췄습니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목표 | 자연어 질문으로 컴퓨터 비전 특허를 검색하고 근거 기반으로 답하는 챗봇 |
| 대상 | 변리사 고용이 부담스러운 개인 발명가 · 중소기업, 특허 실무자의 보조 도구 |
| 접근 | RAG(검색 증강 생성) + LLM Agent(Tool 자동 선택) |
| 데이터 | IPC 코드 약 7만 건, 공개 특허 약 3만 건(청구항 약 59만 건) |
| 산출물 | ① 검색 엔진(`rag_engine`) → ② 웹 챗봇(`web_service`) |

두 축의 검색을 다룹니다.
- **특허 청구항 검색**: 기술 설명 → 유사 특허(청구항) 검색
- **IPC 코드 검색**: 기술 키워드 → 적절한 IPC 분류 코드 추천/설명

---

## 🗂️ 저장소 구조

```
Vision-Patent-Search-LLM-Chatbot
│
├─ rag_engine/                 # [3차] RAG 검색 엔진 (백엔드 코어)
│   ├─ app/                    # LangGraph ReAct 에이전트 + 4개 Tool
│   │   ├─ main.py             #   에이전트 실행 (create_react_agent + 메모리)
│   │   ├─ total_tools.py      #   4개 Tool 정의
│   │   ├─ total_schemas.py    #   Pydantic 입출력 스키마
│   │   ├─ doc_func.py         #   특허 하이브리드 검색 (벡터+BM25)
│   │   └─ ipc_func.py         #   IPC 계층 검색 로직
│   ├─ streamlit/              # Streamlit 데모 UI
│   ├─ db_search/              # ChromaDB 구축·검색 최적화 노트북
│   ├─ doc/                    # 특허 청구항 파이프라인 (preprocessing→embedding→search_upgrade)
│   ├─ ipc/                    # IPC 코드 파이프라인 (전처리·계층화·임베딩)
│   ├─ requirements.txt
│   └─ README.md               # ← rag_engine 상세 설명
│
└─ web_service/                # [4차] Django 웹 서비스
    ├─ config/                 # Django 설정 (settings / urls / wsgi)
    ├─ account/                # 회원가입·로그인·마이페이지·탈퇴
    ├─ chat/                   # 채팅 스트리밍·히스토리 관리 API
    ├─ main/                   # 메인 페이지
    ├─ llm_module/             # rag_engine 코어를 웹용으로 이식
    ├─ templates/  static/     # HTML / CSS / JS
    ├─ requirements.txt
    ├─ manage.py
    └─ README.md               # ← web_service 상세 설명
```

각 하위 폴더의 자세한 설명은 [`rag_engine/README.md`](rag_engine/README.md), [`web_service/README.md`](web_service/README.md) 참고.

> **대용량·비밀 파일 제외**: ChromaDB 벡터 DB(`doc_db/`, `ipc_db/`, 약 11GB), 전처리 데이터(`*.jsonl`), API 키(`.env`), 세션·SQLite 파일은 `.gitignore`로 저장소에서 제외했습니다. 필요한 환경 변수는 [`.env.example`](.env.example) 참고.

---

## 🧠 1) RAG 검색 엔진 (`rag_engine`)

사용자의 자연어 질문을 LLM 에이전트가 분석해, 상황에 맞는 Tool을 스스로 골라 검색을 수행하고 근거 기반으로 답합니다.

### 아키텍처
- **LLM Agent**: LangGraph `create_react_agent` 기반. 질문을 보고 아래 4개 Tool 중 필요한 것을 자동 선택·연쇄 호출
  1. `tool_search_patent_with_description` — 유사 특허(청구항) 검색
  2. `tool_search_detail_patent_by_id` — 출원번호로 특허 단건 조회
  3. `tool_search_ipc_code_with_description` — 기술 키워드 → IPC 후보 추천
  4. `tool_search_ipc_description_from_code` — IPC 코드 → 상세·계층 설명
- **벡터 DB**: ChromaDB
- **임베딩**: IPC = `text-embedding-3-small`, 청구항 = `dragonkue/BGE-m3-ko`

### 검색 최적화 (핵심)
단순 벡터 검색의 한계를 여러 단계로 보완했습니다.

- **청구항 검색 (`doc_func.py`)**
  - 멀티 쿼리 시 쿼리별 거리를 **z-score 정규화**해 특정 쿼리 편향 완화
  - **벡터(0.7) + BM25(0.3) 하이브리드** 검색
  - 청구항 단위가 아닌 **특허 단위로 재집계** (상위 청구항 평균·최고점·개수를 가중 결합)
- **IPC 검색 (`ipc_func.py`)**
  - 거리 컷오프(1.4)로 무관한 코드 제거
  - 상·하위 코드의 의미 중복을 계층 병합으로 해소, 형제 코드는 가장 가까운 하나만 선택

---

## 🌐 2) Django 웹 서비스 (`web_service`)

3차 검색 엔진을 실제 사용할 수 있는 웹 챗봇으로 배포했습니다.

- **회원 / 비회원 지원**: 로그인 없이도 세션 기반으로 대화 가능, 회원가입 시 게스트 대화를 계정으로 이관
- **실시간 스트리밍 응답**: NDJSON 스트리밍으로 토큰 단위 출력, 도구 호출 상태도 함께 표시
- **채팅 관리**: 히스토리 목록, 제목 자동 생성(첫 질문 요약), 드래그 순서 변경, 즐겨찾기, 부분 삭제
- **엔진 재구성**: 3차의 `create_react_agent` → 4차에서 직접 `StateGraph`로 그래프 구성, 메모리는 Django DB 기반으로 전환(`memory_utils.py`가 DB ↔ LangChain 메시지 변환)

기술 스택: Django · LangChain / LangGraph · ChromaDB · OpenAI · Docker / Nginx / Gunicorn(배포)

---

## 🙋‍♀️ 담당한 부분

팀 프로젝트에서 제가 주로 맡은 역할입니다.

**3차 (검색 엔진)**
- 청구항 하이브리드 검색의 **성능 평가 지표 설계** — "같은 IPC 코드를 공유하면 유사 특허"라는 기준으로 정답을 가정하고, 검색 결과와 비교해 개선 효과를 정량화
- ChromaDB 임베딩·검색 실험 (`doc/rag/embedding`, `search_upgrade`)

**4차 (웹 서비스)**
- 프론트엔드(HTML/CSS/JavaScript) 및 화면 디자인
- `account` 앱 — 회원가입·로그인·마이페이지·회원 탈퇴 기능

---

## 💭 회고

이 프로젝트를 정리하며 다시 느낀 것은 **"모델 성능보다 설계와 데이터 흐름이 중요하다"** 는 점이었습니다.

- 좋은 LLM을 붙이는 것만으로는 부족했고, 어떤 Tool을 두고 어떤 스키마로 정보를 주고받을지, 검색 결과를 어떻게 재정렬할지가 답변 품질을 갈랐습니다.
- 하이브리드 검색을 도입했을 때 "정말 나아졌는가"를 말하려면 **객관적인 평가 지표**가 필요했고, 이를 설계하는 과정에서 검색 품질을 수치로 바라보는 법을 배웠습니다.
- 3차의 로컬 검색 엔진을 4차에서 웹으로 옮기며, 메모리 관리와 스트리밍처럼 "서비스로 만들 때 비로소 생기는 문제"들을 다뤘습니다.

### 정리하면서 개선한 점
- 두 프로젝트를 하나의 저장소로 통합하고, 폴더를 의미 기반(`rag_engine` / `web_service`)으로 재배치
- 이중 중첩 폴더(`Pai-Django/_pai/_pai`)를 평탄화하고 설정 폴더를 `config`로 정리
- 11GB에 달하는 벡터 DB와 API 키를 `.gitignore`로 분리해, 저장소는 코드만 담도록 정리
- 작업 잔여물(병합 충돌이 남은 파일, 세션 캐시, 중복 이미지)을 제거
- 하드코딩돼 있던 Django `SECRET_KEY`를 환경변수(`.env`)로 분리하고, 중복 정의된 세션 설정을 정리
- 프로젝트별 `requirements.txt`와 하위 `README`를 추가해 재현·이해가 쉽도록 보완

---

## 🚀 실행

두 프로젝트는 독립적으로 실행됩니다. 자세한 절차는 각 하위 README를 참고하세요.

```bash
# 공통: 환경 변수 준비
cp .env.example rag_engine/.env      # 값 채우기 (OPENAI_API_KEY 등)
cp .env.example web_service/.env     # 값 채우기 (+ DJANGO_SECRET_KEY)

# RAG 검색 엔진
cd rag_engine && pip install -r requirements.txt
streamlit run streamlit/app.py

# 웹 서비스
cd web_service && pip install -r requirements.txt
python manage.py migrate && python manage.py runserver
```

> 벡터 DB(`doc_db/`, `ipc_db/`)는 대용량이라 저장소에 없습니다. `rag_engine`의 구축 노트북으로 생성한 뒤, `web_service/db_search/` 아래에 배치하면 웹 서비스가 이를 참조합니다.
