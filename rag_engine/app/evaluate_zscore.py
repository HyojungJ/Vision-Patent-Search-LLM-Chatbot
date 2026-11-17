"""
[Z-score 전후 비교] 같은 질문 30개로 Hit@5, Hit@10 을 측정한다.

- Before: 하위 쿼리 결과를 원래 거리로 합침  (patent_hybrid_search(..., merge="distance"))
- After : 하위 쿼리 결과를 Z-score로 합침     (patent_hybrid_search(..., merge="zscore"), 현재 기본값)
- 나머지 조건(후보 200개, BM25 혼합 0.7:0.3, 특허 단위 점수)은 완전히 같다.
- Hit@k: 상위 k개 특허 안에 정답 특허가 하나 이상 있는 질문의 비율

사용: (rag_engine/app 폴더에서) python evaluate_zscore.py
입력: zscore_eval_queries.json   결과: zscore_eval_results.json
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from doc_func import patent_hybrid_search

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "db_search", "doc_db")
Q_PATH = os.path.join(HERE, "zscore_eval_queries.json")
OUT_PATH = os.path.join(HERE, "zscore_eval_results.json")
MODES = {"Before (거리 그대로)": "distance", "After (Z-score)": "zscore"}


def main():
    t0 = time.time()
    queries = json.load(open(Q_PATH, encoding="utf-8"))["queries"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("dragonkue/BGE-m3-ko").to(device)
    col = chromadb.PersistentClient(path=DB_PATH).get_collection("patent_claims")
    print(f"준비 완료 ({time.time() - t0:.0f}초), 질문 {len(queries)}개", flush=True)

    per_query = []
    for i, q in enumerate(queries, 1):
        row = {"qid": q["qid"], "gold": q["gold_patents"]}
        for label, mode in MODES.items():
            res = patent_hybrid_search(
                collection=col, model=model, query_list=q["sub_queries"],
                per_query_top_k=200, final_top_k=200, top_k=10, merge=mode,
            )
            ranked = [r["patent_id"] for r in res]
            hits = [ranked.index(g) + 1 for g in q["gold_patents"] if g in ranked]
            row[mode] = min(hits) if hits else None  # 정답 최고 순위 (10위 밖이면 None)
        per_query.append(row)
        print(f"{i}/{len(queries)} {q['qid']} distance={row['distance']} zscore={row['zscore']} "
              f"({time.time() - t0:.0f}초)", flush=True)

    summary = {}
    for label, mode in MODES.items():
        ranks = [r[mode] for r in per_query]
        summary[label] = {
            "Hit@5": sum(1 for x in ranks if x is not None and x <= 5) / len(ranks),
            "Hit@10": sum(1 for x in ranks if x is not None and x <= 10) / len(ranks),
        }
    json.dump({"summary": summary, "per_query": per_query}, open(OUT_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for label, s in summary.items():
        print(f"{label}: Hit@5 {s['Hit@5'] * 100:.1f}%  Hit@10 {s['Hit@10'] * 100:.1f}%", flush=True)
    print(f"끝 ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
