# 🧠 rag_engine — 특허 검색 RAG 엔진

컴퓨터 비전 특허 질의응답 챗봇의 **검색·추론 코어**. 자연어 질문을 LLM 에이전트가 분석해 상황에 맞는 Tool을 스스로 선택하고, ChromaDB에서 특허·IPC를 검색해 근거 기반으로 답합니다.

이 엔진은 이후 [`../web_service`](../web_service)에서 Django 웹 서비스로 배포됩니다.

---

## 구조

```
rag_engine
├─ app/                        # LangGraph 에이전트 (핵심 실행부)
│   ├─ main.py                 #   ReAct 에이전트 실행 + 대화 메모리
│   ├─ total_tools.py          #   4개 Tool 정의 (@tool)
│   ├─ total_schemas.py        #   Pydantic 입출력 스키마
│   ├─ doc_func.py             #   특허 청구항 하이브리드 검색
│   └─ ipc_func.py             #   IPC 계층 검색 로직
│
├─ streamlit/                  # Streamlit 데모 UI (app/ 과 동일 로직 + 웹 화면)
│
├─ db_search/                  # ChromaDB 구축·검색 최적화 노트북
│   ├─ fixed_final.ipynb       #   최종 DB 구축
│   └─ search_optimizing.ipynb #   검색 파라미터 튜닝
│   └─ (doc_db/, ipc_db/)      #   ← 실제 벡터 DB. .gitignore 제외 (약 11GB)
│
├─ doc/                        # 특허 청구항 데이터 파이프라인
│   └─ rag/
│       ├─ preprocessing/      #   XML 파싱 → 청구항 분리 (xml_read, data_preprocessing)
│       ├─ embedding/          #   임베딩 생성·평가 (embedding_final, evalution_test)
│       └─ search_upgrade/     #   하이브리드 검색 고도화 실험
│
└─ ipc/                        # IPC 코드 데이터 파이프라인
    ├─ data/                   #   IPC 원본·전처리 (raw → processed_data v1~v5)
    └─ rag/                    #   IPC 벡터 DB 구축 (build_chroma_clean)
```

> **폴더가 여러 벌인 이유**: `app/`은 순수 실행 로직, `streamlit/`은 같은 로직에 데모 화면을 얹은 버전입니다. 개발 과정상 두 벌이 존재하며, 핵심 로직(`doc_func`, `ipc_func`, `total_tools`, `total_schemas`)은 동일합니다.

---

## 에이전트 & Tool

**LangGraph `create_react_agent`** 기반. LLM이 질문을 보고 아래 4개 Tool 중 필요한 것을 자동으로 선택·연쇄 호출합니다.

| Tool | 역할 |
|------|------|
| `tool_search_patent_with_description` | 기술 설명 → 유사 특허(청구항) 검색 |
| `tool_search_detail_patent_by_id` | 출원번호 → 특허 단건 상세 조회 |
| `tool_search_ipc_code_with_description` | 기술 키워드 → IPC 후보 코드 추천 |
| `tool_search_ipc_description_from_code` | IPC 코드 → 상세 설명·계층 구조 |

- **임베딩**: IPC = `text-embedding-3-small`(OpenAI), 청구항 = `dragonkue/BGE-m3-ko`(로컬)
- **입출력 스키마**: 모든 Tool은 Pydantic 스키마로 입출력을 구조화. 검색 입력에 `exclude_patent_ids`, 결과에 `result_index`를 두어 "N번 결과 빼고 다시 검색" 같은 후속 요청을 지원.

---

## 검색 최적화

단순 벡터 유사도 검색의 한계를 단계적으로 보완했습니다.

### 특허 청구항 검색 (`doc_func.py`)
1. **멀티 쿼리 z-score 정규화** — 쿼리가 여러 개일 때 쿼리별 거리 분포를 z-score로 정규화해, 특정 쿼리에 결과가 쏠리는 편향을 완화
2. **하이브리드 검색** — 벡터 유사도(0.7) + BM25 키워드 점수(0.3) 결합
3. **특허 단위 재집계** — 청구항이 아닌 특허 단위로 점수 산정 (상위 3개 청구항 평균 0.6 + 최고점 0.3 + 청구항 수 보너스 0.1)

### IPC 코드 검색 (`ipc_func.py`)
1. **거리 컷오프(1.4)** — 무관한 코드를 노이즈로 간주해 제거
2. **계층 병합** — 상·하위 코드의 의미 중복을 병합해 단절 해소
3. **형제 코드 중복 제거** — 같은 부모를 가진 코드는 가장 가까운 하나만 선택

> 성능 평가는 "같은 IPC 코드를 공유하면 유사 특허"라는 가정을 정답 기준으로 삼아, 검색 결과에 정답 IPC가 얼마나 포함되는지로 정량화했습니다.

---

## 실행

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 환경 변수 설정 (.env)
#    루트의 .env.example 참고 (OPENAI_API_KEY 등)

# 3) 벡터 DB 준비
#    db_search/ 및 ipc/rag 의 노트북을 실행해 doc_db/, ipc_db/ 생성
#    (원본 데이터·DB는 대용량이라 저장소에서 제외됨)

# 4) 실행
python app/main.py              # CLI 에이전트
streamlit run streamlit/app.py  # 데모 UI
```

---

## 기술 스택
Python · LangChain / LangGraph · ChromaDB · OpenAI · SentenceTransformers · BM25 · Streamlit
