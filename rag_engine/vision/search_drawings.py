"""
[유사 특허 도면 검색 - OpenCV]

쿼리 도면 1장을 주면, images 폴더의 다른 도면들 중 '구조가 비슷한' 순서로 top-k를 돌려준다.
딥러닝 없이 OpenCV의 ORB 특징점 매칭만으로 유사도를 계산한다.

유사도 계산 방법:
  1. 두 도면에서 ORB 디스크립터를 뽑는다 (drawing_features.extract_features).
  2. BFMatcher(해밍 거리) + knnMatch(k=2)로 쿼리의 각 특징점에 대해
     상대 도면에서 가장 가까운 2개를 찾는다.
  3. Lowe's ratio test: 1등 거리가 2등 거리의 0.75배보다 작으면 '확실한 매칭'으로 인정.
  4. '좋은 매칭' 개수를 정규화해 유사도 점수로 삼는다.
     score = good_matches / min(쿼리 특징점수, 후보 특징점수)
     -> 특징점이 적은 도면에서도 공정하게 비교되도록 분모를 min으로.

사용 예:
  python search_drawings.py 1020160171671.jpg          # 폴더 내 도면을 쿼리로
  python search_drawings.py path/to/query.png --topk 3 # 외부 이미지를 쿼리로
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import cv2

from drawing_features import extract_features

# Lowe's ratio test 임계값 (작을수록 엄격)
RATIO_THRESHOLD = 0.75

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 출원번호 -> 특허 제목 매핑 (전처리 데이터에서 로드)
_TITLE_MAP = None


def load_title_map():
    """
    preprocessed_data jsonl에서 '출원번호 -> 제목' 매핑을 1회 로드.
    이미지 검색 결과(출원번호)에 사람이 읽을 수 있는 제목을 붙이기 위함.
    (파일명이 출원번호이므로, 번호를 다리로 기존 특허 데이터와 연결된다.)
    """
    global _TITLE_MAP
    if _TITLE_MAP is not None:
        return _TITLE_MAP

    _TITLE_MAP = {}
    base = os.path.dirname(__file__)
    jsonl = os.path.join(
        base, "..", "doc", "data", "preprocessed_data", "preprocessed_data_100.jsonl"
    )
    try:
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pid = d.get("id")
                name = (d.get("embed") or {}).get("name", "")
                if pid:
                    _TITLE_MAP[pid] = name
    except FileNotFoundError:
        pass  # 제목 파일이 없으면 번호만 표시
    return _TITLE_MAP


def count_good_matches(desc1, desc2) -> int:
    """두 디스크립터 집합 간 Lowe ratio test를 통과한 '좋은 매칭' 수."""
    if desc1 is None or desc2 is None:
        return 0
    if len(desc1) < 2 or len(desc2) < 2:
        return 0

    # ORB 디스크립터는 이진(binary) -> 해밍 거리
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_THRESHOLD * n.distance:
            good += 1
    return good


def similarity_score(desc_query, n_query, desc_cand, n_cand):
    """good match 개수를 특징점 수로 정규화한 유사도 점수(0~1 근처)."""
    good = count_good_matches(desc_query, desc_cand)
    denom = max(1, min(n_query, n_cand))
    return good / denom, good


def list_drawings(img_dir: str):
    """폴더 내 이미지 파일 경로 리스트."""
    return [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.lower().endswith(IMAGE_EXTS)
    ]


def search(query_path: str, img_dir: str, topk: int = 5):
    """
    query_path 도면과 img_dir 안의 다른 도면들을 비교해 유사도 top-k 반환.
    반환: list[dict(patent_id, file, score, good_matches)]
    """
    # 쿼리 특징
    _, q_desc, q_info = extract_features(query_path)
    q_n = q_info["num_keypoints"]
    query_name = os.path.basename(query_path)
    title_map = load_title_map()

    results = []
    for cand_path in list_drawings(img_dir):
        cand_name = os.path.basename(cand_path)
        # 쿼리 자신은 제외 (파일명 기준)
        if os.path.abspath(cand_path) == os.path.abspath(query_path):
            continue

        _, c_desc, c_info = extract_features(cand_path)
        score, good = similarity_score(q_desc, q_n, c_desc, c_info["num_keypoints"])

        pid = os.path.splitext(cand_name)[0]
        results.append({
            "patent_id": pid,
            "title": title_map.get(pid, ""),   # 출원번호 -> 제목 연결
            "file": cand_name,
            "score": round(score, 4),
            "good_matches": good,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return query_name, results[:topk]


def main():
    base = os.path.dirname(__file__)
    default_dir = os.path.join(base, "..", "images")

    parser = argparse.ArgumentParser(description="OpenCV 기반 유사 특허 도면 검색")
    parser.add_argument("query", help="쿼리 이미지 파일명(images 폴더 내) 또는 전체 경로")
    parser.add_argument("--dir", default=default_dir, help="도면 폴더 (기본: ../images)")
    parser.add_argument("--topk", type=int, default=5, help="상위 몇 개 반환 (기본 5)")
    args = parser.parse_args()

    # 쿼리 경로 해석: 폴더 내 파일명이면 폴더와 합침
    query_path = args.query
    if not os.path.isfile(query_path):
        query_path = os.path.join(args.dir, args.query)
    if not os.path.isfile(query_path):
        print(f"[ERROR] 쿼리 이미지를 찾을 수 없습니다: {args.query}")
        return

    query_name, top = search(query_path, args.dir, args.topk)

    print(f"\n쿼리 도면: {query_name}")
    print(f"유사 도면 top-{len(top)} (OpenCV ORB 특징점 매칭 기준)\n")
    for i, r in enumerate(top, 1):
        title = r["title"] or "(제목 정보 없음)"
        print(f"{i:>2}. [{r['score']:.4f}] {r['patent_id']}  (매칭 {r['good_matches']}개)")
        print(f"     {title}")


if __name__ == "__main__":
    main()
