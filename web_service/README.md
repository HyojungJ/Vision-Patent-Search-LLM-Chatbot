# 🌐 web_service — Pai Django 웹 챗봇

[`../rag_engine`](../rag_engine)의 특허 검색 엔진을 실제 사용할 수 있는 **웹 챗봇 서비스**로 배포한 Django 프로젝트입니다. 서비스명은 **Pai**(Patent AI).

---

## 구조

```
web_service
├─ config/                 # Django 프로젝트 설정
│   ├─ settings.py         #   설정 (SECRET_KEY는 환경변수로 분리)
│   ├─ urls.py             #   루트 URL 라우팅
│   ├─ wsgi.py / asgi.py
│
├─ account/                # 회원 관리 앱
│   ├─ views.py            #   회원가입·로그인·마이페이지·탈퇴
│   ├─ forms.py            #   가입/로그인/프로필 폼
│   └─ models.py
│
├─ chat/                   # 채팅 앱 (핵심)
│   ├─ views.py            #   스트리밍 응답 + 히스토리 관리 API
│   ├─ models.py           #   ChatHistory / Chat 모델
│   └─ urls.py
│
├─ main/                   # 메인 페이지 앱
│
├─ llm_module/             # rag_engine 코어를 웹용으로 이식
│   ├─ main.py             #   StateGraph 에이전트 (웹용 재구성)
│   ├─ total_tools.py      #   4개 Tool (DB 경로를 Django BASE_DIR 기준으로 조정)
│   ├─ doc_func.py         #   특허 하이브리드 검색
│   ├─ ipc_func.py         #   IPC 계층 검색
│   ├─ memory_utils.py     #   DB ↔ LangChain 메시지 변환
│   └─ SYSTEM_PROMPT.py    #   시스템 프롬프트 (별도 파일로 분리)
│
├─ db_search/              # ← 실행 시 참조하는 벡터 DB 위치 (doc_db/, ipc_db/)
│                          #   .gitignore 제외. rag_engine에서 구축한 DB를 여기 배치
│
├─ templates/  static/     # HTML / CSS / JS
└─ manage.py
```

> **`llm_module`은 왜 따로 있나**: `rag_engine/app`의 검색 로직을 그대로 가져오되, 웹 환경에 맞게 (1) import를 상대경로로, (2) DB 경로를 `settings.BASE_DIR/db_search` 기준으로, (3) 에이전트를 `create_react_agent` → 직접 `StateGraph`로 재구성했습니다.

---

## 주요 기능

### 채팅 (`chat`)
- **실시간 스트리밍**: `StreamingHttpResponse` + NDJSON으로 토큰 단위 출력. 도구 호출 상태(어떤 Tool이 실행 중인지)도 함께 전송
- **대화 메모리**: 대화 내역을 DB에 저장하고, 후속 질문 시 이전 맥락을 불러와 LLM에 전달 (`memory_utils.py`)
- **비동기 제목 생성**: 첫 질문을 요약해 채팅방 제목 자동 생성 (`ThreadPoolExecutor`로 응답 생성과 병렬 처리)
- **중간 저장**: 스트리밍 도중 1.5초마다 응답을 DB에 체크포인트 저장 (중단 대비)
- **히스토리 관리**: 목록 조회, 드래그 순서 변경(`bulk_update` 최적화), 즐겨찾기, 부분 삭제

### 회원 (`account`)
- 회원가입 / 로그인 / 마이페이지(닉네임·비밀번호 변경) / 회원 탈퇴
- **게스트 지원**: 로그인 없이 세션 기반으로 대화 가능
- **게스트 → 회원 이관**: 회원가입 시 게스트로 나눈 대화를 새 계정으로 자동 이전

### URL 요약
| 경로 | 기능 |
|------|------|
| `/` | 메인 페이지 |
| `/account/` | 회원가입·로그인·마이페이지 |
| `/chat/chat/` | 채팅 인터페이스 |
| `/chat/api/stream/` | 채팅 스트리밍 API |
| `/chat/api/history/*` | 히스토리 관리 (순서·이름·삭제·즐겨찾기) |

---

## 실행

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 환경 변수 (.env) — 루트 .env.example 참고
#    OPENAI_API_KEY, DJANGO_SECRET_KEY 등

# 3) 벡터 DB 배치
#    rag_engine에서 구축한 doc_db/, ipc_db/ 를 db_search/ 아래에 위치

# 4) DB 마이그레이션 & 실행
python manage.py migrate
python manage.py runserver
```

---

## 기술 스택
Django · LangChain / LangGraph · ChromaDB · OpenAI · SentenceTransformers · Docker / Nginx / Gunicorn(배포)
