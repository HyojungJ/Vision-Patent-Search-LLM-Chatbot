"""
[Z-score 평가용] 검색 질문과 정답 특허 목록을 만든다.

질문 만들기:
  - 원본 XML 28,079건 중 무작위로 30건을 고른다 (시드 고정 -> 매번 같은 30건).
  - 각 특허의 초록(astrtCont)을 '...다.' 단위 문장으로 나눈다.
  - 제목을 되풀이하는 첫 문장("~가 개시된다." 등)은 뺀다.
  - 남은 문장 중 앞의 3개를 하위 쿼리로 쓴다. (한 아이디어를 여러 기술 요소로 나눠 묻는 상황)
  - 문장이 3개 미만인 특허는 후보에서 뺀다.

정답 만들기:
  - 질문을 만든 그 특허.
  - 초록이 글자 그대로 같은 다른 특허(같은 계열 출원)도 정답에 포함한다.

한계: 질문이 초록 문장이라 실제 사용자 질문보다 쉬운 편이다.

사용: (rag_engine/app 폴더에서) python build_zscore_queries.py
결과: zscore_eval_queries.json
"""

import html
import json
import os
import random
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "doc", "data", "raw_data")
OUT_PATH = os.path.join(HERE, "zscore_eval_queries.json")

N_QUERIES = 30
N_SUB = 3            # 질문 하나당 하위 쿼리 수
MIN_LEN = 20         # 이보다 짧은 문장은 버림
MAX_LEN = 300        # 너무 긴 문장은 앞부분만 사용
SEED = 42

ABSTRACT_RE = re.compile(r"<astrtCont>(.*?)</astrtCont>", re.S)
TITLE_RE = re.compile(r"<inventionTitle>(.*?)</inventionTitle>", re.S)
BOILERPLATE = re.compile(r"(개시된다|개시한다|제공된다|제공한다|관한 것이다)\.$")


def clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(abstract: str):
    sents = [s.strip() for s in re.findall(r".+?다\.", abstract)]
    if sents and BOILERPLATE.search(sents[0]):
        sents = sents[1:]
    return [s[:MAX_LEN] for s in sents if len(s) >= MIN_LEN]


def main():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".xml"))
    abstracts, titles = {}, {}
    for f in files:
        pid = f[:-4]
        x = open(os.path.join(RAW_DIR, f), encoding="utf-8").read()
        m = ABSTRACT_RE.search(x)
        t = TITLE_RE.search(x)
        abstracts[pid] = clean(m.group(1)) if m else ""
        titles[pid] = clean(t.group(1)) if t else ""

    # 초록이 똑같은 특허끼리 묶기 (정답 목록에 함께 넣음)
    same_abstract = defaultdict(list)
    for pid, a in abstracts.items():
        if a:
            same_abstract[a].append(pid)

    candidates = []
    for pid in sorted(abstracts):
        sents = split_sentences(abstracts[pid])
        if len(sents) >= N_SUB:
            candidates.append((pid, sents[:N_SUB]))

    picked = random.Random(SEED).sample(candidates, N_QUERIES)

    queries = []
    for i, (pid, subs) in enumerate(sorted(picked), 1):
        gold = sorted(same_abstract[abstracts[pid]])
        queries.append({
            "qid": f"Q{i:02d}",
            "source_patent": pid,
            "title": titles[pid],
            "sub_queries": subs,
            "gold_patents": gold,
        })

    meta = {
        "n_patents_scanned": len(files),
        "n_candidates_with_3plus_sentences": len(candidates),
        "n_queries": len(queries),
        "seed": SEED,
        "n_queries_with_multiple_gold": sum(len(q["gold_patents"]) > 1 for q in queries),
    }
    json.dump({"meta": meta, "queries": queries}, open(OUT_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
