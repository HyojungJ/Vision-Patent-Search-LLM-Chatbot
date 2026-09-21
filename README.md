# 🔎 컴퓨터 비전 특허 검색·질의응답 RAG 챗봇 서비스

자연어 질문을 분석해 **유사 특허와 IPC 분류 코드를 검색**하고, 검색 근거를 바탕으로 답변하는 RAG 챗봇 서비스입니다.
검색 엔진을 먼저 구축한 뒤 Django 웹 서비스로 확장했으며, 프로젝트 종료 후 전체 구조를 개인 포트폴리오 기준으로 리팩토링하고 복기했습니다.

> **프로젝트 성격**
> SKN 부트캠프 팀 프로젝트(5인)로 진행한 두 단계의 연속 프로젝트입니다.
> **1단계: 특허 검색 엔진 구축(부트캠프 3차) → 2단계: Django 웹 서비스 확장(4차)** 순으로 개발했으며,
> 이후 개인적으로 폴더 구조, 설정, 문서와 보안 항목을 정리했습니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| 목표 | 자연어 질문으로 컴퓨터 비전 특허·IPC 코드를 검색하고 근거 기반 답변 제공 |
| 대상 사용자 | 개인 발명가, 중소기업, 특허 실무자의 초기 선행기술 조사 보조 |
| 접근 방식 | RAG + LLM Agent의 Tool 자동 선택·연쇄 호출 |
| 데이터 | IPC 코드 약 7만 건, 공개 특허 약 3만 건, 청구항 약 59만 건 |
| 검색 엔진 | ChromaDB 벡터 검색 + BM25 하이브리드 검색 + 멀티 쿼리 Z-score 정규화 |
| 웹 서비스 | Django 기반 회원·비회원 채팅, NDJSON 실시간 스트리밍 |
| 사용 기술 | Python, LangChain, LangGraph, ChromaDB, OpenAI API, Django, HTML/CSS/JavaScript, Docker, Nginx, Gunicorn |

---

## 🗂️ 저장소 구조

```
Vision-Patent-Search-LLM-Chatbot
│
├─ rag_engine/                 # 특허·IPC RAG 검색 엔진
│   ├─ app/                    # LangGraph ReAct Agent + Tool
│   │   ├─ main.py             #   에이전트 실행 및 대화 메모리
│   │   ├─ total_tools.py      #   특허·IPC 검색 Tool 4종
│   │   ├─ total_schemas.py    #   Pydantic 입출력 스키마
│   │   ├─ doc_func.py         #   벡터+BM25 특허 하이브리드 검색
│   │   └─ ipc_func.py         #   IPC 계층 검색·중복 제거
│   ├─ streamlit/              # 검색 엔진 데모 UI
│   ├─ db_search/              # ChromaDB 구축·검색 최적화 노트북
│   ├─ doc/                    # 특허 청구항 전처리·임베딩·검색 실험
│   ├─ ipc/                    # IPC 코드 전처리·계층화·임베딩
│   ├─ requirements.txt
│   └─ README.md
│
└─ web_service/                # Django 웹 챗봇 서비스
    ├─ config/                 # Django 설정·URL·WSGI/ASGI
    ├─ account/                # 회원가입·로그인·마이페이지·회원 탈퇴
    ├─ chat/                   # 스트리밍 채팅·대화 내역 관리 API
    ├─ main/                   # 메인 페이지
    ├─ llm_module/             # RAG 엔진을 웹 서비스용으로 이식
    ├─ templates/              # Django HTML 템플릿
    ├─ static/                 # CSS·JavaScript·서비스 이미지
    ├─ requirements.txt
    ├─ manage.py
    └─ README.md
```

각 하위 프로젝트의 상세 내용은 [`rag_engine/README.md`](rag_engine/README.md)와
[`web_service/README.md`](web_service/README.md)를 참고할 수 있습니다.

---

## 🛠️ 기술 스택

- **언어**: Python, HTML, CSS, JavaScript
- **LLM / Agent**: OpenAI API, LangChain, LangGraph
- **검색 / RAG**: ChromaDB, SentenceTransformers, BM25, 멀티 쿼리 검색
- **웹**: Django, Streamlit
- **데이터 검증**: Pydantic
- **배포**: Docker, Nginx, Gunicorn

---

## 1. 개요

자연어로 질문하면 유사 특허와 IPC 분류 코드를 찾아 근거를 들어 답하는 RAG 챗봇을 만들었습니다.
변리사를 바로 선임하기 부담스러운 개인 발명가나 중소기업이 초기 선행기술을 탐색하고,
특허 실무자가 보조 도구로 활용할 수 있도록 설계했습니다.

먼저 컴퓨터 비전 특허·IPC 데이터를 검색하는 RAG 엔진을 구축했고,
이후 회원 관리와 대화 기록, 실시간 스트리밍을 지원하는 Django 웹 서비스로 확장했습니다.

두 축의 검색을 제공합니다.
- **특허 청구항 검색**: 기술 설명 → 의미가 유사한 청구항 검색 → 특허 단위 재정렬
- **IPC 코드 검색**: 기술 키워드 → IPC 후보 추천 또는 입력된 IPC 코드의 계층·의미 설명

---

## 2. 담당 역할과 문제 해결 과정

### 1) 검색 성능 평가 지표 설계 — 검색 엔진 단계

하이브리드 검색을 도입했을 때 "정말 검색 품질이 좋아졌는가"를 판단할 객관적인 기준이 필요했습니다.
이를 위해 **같은 IPC 코드를 공유하면 유사한 특허**라는 도메인 기준으로 정답셋을 가정하고,
검색 결과 특허의 IPC 코드가 정답 IPC를 얼마나 포함하는지 비교했습니다.

이 기준을 바탕으로 다음 검색 전략의 적용 전후를 수치로 검증했습니다.
- 벡터 유사도와 BM25를 **0.7 : 0.3**으로 결합한 하이브리드 검색
- 여러 쿼리의 거리 분포 차이를 보정하는 **멀티 쿼리 Z-score 정규화**
- 청구항 결과를 출원번호 기준으로 묶어 다시 계산하는 **특허 단위 리랭킹**

### 2) ChromaDB 임베딩 및 검색 실험 — 검색 엔진 단계

청구항과 IPC 코드는 데이터의 성격이 달라 각각 별도의 검색 파이프라인을 구성했습니다.

- 특허 청구항: 한국어 의미 검색에 적합한 `dragonkue/BGE-m3-ko` 임베딩 사용
- IPC 코드: `text-embedding-3-small`로 IPC 설명을 임베딩
- 검색 개수, 거리 임계값, 하이브리드 가중치와 특허 집계 점수를 반복 실험
- IPC 상·하위 코드의 의미 중복과 형제 코드 오염 문제를 줄이기 위한 계층 기반 후처리 적용

### 3) 프론트엔드 및 화면 구현 — 웹 서비스 단계

HTML/CSS/JavaScript로 웹 서비스 화면을 디자인하고 구현했습니다.

- NDJSON 방식으로 수신한 토큰을 실시간으로 화면에 추가하는 스트리밍 채팅 UI
- LLM이 어떤 Tool을 호출하고 있는지 보여주는 도구 실행 상태 UI
- 대화방 목록, 제목 수정, 드래그 순서 변경, 즐겨찾기, 메시지 삭제 인터페이스
- 회원·비회원 상태에 따른 화면과 사용자 흐름 구성

### 4) 회원 시스템 구현 — 웹 서비스 단계

회원가입, 로그인, 마이페이지, 회원 탈퇴를 담당하는 Django `account` 앱을 구현했습니다.

- Django 인증 시스템 기반 회원가입·로그인·로그아웃
- 닉네임 및 비밀번호 변경
- 계정 탈퇴 시 관련 데이터 정리
- 로그인하지 않은 사용자를 위한 세션 기반 게스트 채팅
- 게스트가 회원가입하면 기존 세션 대화를 새 계정으로 자동 이관

### 5) 아키텍처 통합 및 리팩토링 — 개인 복기

프로젝트 종료 후 두 결과물을 하나의 포트폴리오 저장소로 통합하고 구조를 정리했습니다.

- 이중으로 중첩된 팀 저장소 폴더를 `rag_engine`과 `web_service` 중심으로 평탄화
- Django 설정 패키지를 `config`로 정리하고 import 경로 갱신
- 3차의 LangGraph 메모리 구조를 4차의 Django DB 대화 기록과 연결하는 흐름 정리
  - DB의 `Chat` 기록 → LangChain 메시지 객체 변환
  - 스트리밍 최종 응답 → DB 저장
- 하드코딩된 Django `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`를 환경변수로 분리
- 약 11GB의 ChromaDB와 API 키·세션·SQLite 파일을 `.gitignore`로 제외해 코드 중심 저장소로 정리

---

## 3. 성과

- **하이브리드 검색 및 Z-score 정규화**를 적용하고 IPC 기반 평가 기준으로 검색 품질 개선을 정량 확인
- 자연어 기술 설명, IPC 코드, 출원번호 등 질문 유형에 따라 적절한 Tool을 선택하는 **4종 검색 Agent** 구축
- 회원·비회원 모두 이용 가능한 **실시간 스트리밍 RAG 챗봇 서비스** 구현
- 게스트 대화 이관, 대화 메모리, 히스토리 관리 기능을 포함한 실제 웹 사용 흐름 완성
- 검색 엔진과 웹 서비스를 재현 가능하고 이해하기 쉬운 단일 포트폴리오 구조로 리팩토링

---

## 4. 구현 구조

### RAG Agent의 4개 Tool

| Tool | 역할 |
|------|------|
| `tool_search_patent_with_description` | 기술 설명으로 유사 특허·청구항 검색 |
| `tool_search_detail_patent_by_id` | 특정 출원번호의 특허 메타데이터·청구항 직접 조회 |
| `tool_search_ipc_code_with_description` | 기술 설명으로 IPC 후보 코드 추천 |
| `tool_search_ipc_description_from_code` | 입력된 IPC 코드의 의미와 계층 구조 조회 |

### 특허 검색 최적화

1. 쿼리별 벡터 검색 결과 수집
2. 멀티 쿼리의 거리 분포를 Z-score로 정규화
3. 벡터 유사도 0.7 + BM25 점수 0.3 결합
4. 청구항을 출원번호별로 그룹화
5. 상위 청구항 평균, 최고 점수, 검색된 청구항 수를 가중 결합해 특허 단위 점수 계산
6. 제외할 특허 ID를 제거하고 최종 결과에 일관된 결과 번호 부여

### IPC 검색 최적화

- 거리 1.4 이상인 검색 결과를 노이즈로 처리
- 상위·하위 코드가 함께 검색되면 더 상세한 하위 코드를 우선하되 조상 정보도 함께 제공
- 동일한 부모를 가진 형제 코드는 가장 가까운 코드만 선택
- 여러 기술 쿼리의 결과를 라운드 로빈 방식으로 결합해 한 쿼리로의 편향 완화

### Django 웹 서비스

- `StreamingHttpResponse`와 NDJSON을 이용한 토큰 스트리밍
- 회원은 사용자 ID, 비회원은 세션 ID로 대화방 소유권 분리
- Django DB의 과거 대화를 LangChain 메시지로 변환해 후속 질문의 문맥 유지
- 첫 질문 기반 대화방 제목 생성을 별도 스레드로 처리
- 스트리밍 중 1.5초마다 응답을 중간 저장해 중단 상황 대비
- `bulk_update`를 활용해 대화방 드래그 순서 변경의 DB 쿼리 최적화

---

## 5. 회고

이 프로젝트를 다시 정리하면서 느낀 건, **좋은 LLM을 붙이는 것만으로는 충분하지 않다**는 점이었습니다.
어떤 Tool을 두고 어떤 스키마로 정보를 주고받을지, 검색 결과를 어떻게 다시 정렬할지가 답변의 품질과 신뢰도를 갈랐습니다.

하이브리드 검색이 "정말 좋아졌는가"를 말하려면 객관적인 평가 기준이 필요했습니다.
IPC 코드 공유 여부를 평가 기준으로 설계하면서 검색 품질을 수치로 바라보는 방법을 배웠고,
벡터 검색과 키워드 검색의 장단점을 함께 활용하는 과정도 경험했습니다.

또한 로컬에서 동작하던 검색 엔진을 웹 서비스로 옮기면서,
대화 메모리, 실시간 스트리밍, 회원·비회원 데이터 분리처럼 **서비스가 되었을 때 비로소 드러나는 문제**를 다룰 수 있었습니다.
프로젝트가 끝난 뒤 구조와 설정을 다시 정리하는 과정에서는 기능 구현뿐 아니라,
다른 사람이 빠르게 이해하고 안전하게 다룰 수 있는 저장소를 만드는 것도 개발의 일부라는 점을 배웠습니다.

---

## 🚀 실행 방법

두 프로젝트는 독립적으로 실행할 수 있습니다. 자세한 절차는 각 하위 README를 참고하세요.

### 1) 환경 변수 준비

```powershell
Copy-Item .env.example rag_engine/.env
Copy-Item .env.example web_service/.env
```

생성한 `.env`에 `OPENAI_API_KEY`, `DJANGO_SECRET_KEY` 등 필요한 값을 입력합니다.

### 2) RAG 검색 엔진

```powershell
Set-Location rag_engine
pip install -r requirements.txt
streamlit run streamlit/app.py
```

### 3) Django 웹 서비스

```powershell
Set-Location web_service
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

> 벡터 DB(`doc_db/`, `ipc_db/`)는 약 11GB의 대용량 실행 데이터라 저장소에 포함하지 않았습니다.
> `rag_engine`의 DB 구축 노트북으로 생성한 뒤 `web_service/db_search/` 아래에 배치해야 웹 서비스가 이를 참조할 수 있습니다.
