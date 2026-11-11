"""
[변형 테스트] 도면을 망가뜨려도 원본을 찾아내는지 측정한다.

방법:
  1. images/ 의 도면 한 장을 골라 일부러 변형한다 (기울임, 축소, 노이즈 ...).
  2. 변형본을 쿼리로 넣어 images/ 전체(원본 포함)를 검색한다.
  3. 원본이 몇 위로 나오는지 기록한다.
  4. 모든 도면에 반복해서 Top-1 정확도, Hit@5 를 계산한다.

지표 정의:
  - 정답: 쿼리의 원본 도면. 파일 내용이 완전히 같은 도면도 같은 정답으로 본다.
  - 순위: 정답이 아닌 도면 중 원본과 점수가 같거나 더 높은 도면의 수 + 1 (동점은 불리하게 계산)
  - Top-1 정확도: 순위가 1위인 쿼리의 비율
  - Hit@5: 순위가 5위 이내인 쿼리의 비율 (상위 5개 안에 정답이 있는가)

주의: 이 테스트는 '같은 도면이 변형돼도 알아보는가'를 잰다.
      서로 다른 특허끼리 비슷한지는 이 테스트로 알 수 없다.

사용: python evaluate_robustness.py
결과: 화면 출력 + eval_results.csv
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import hashlib
import tempfile
import time

import cv2
import numpy as np

from drawing_features import extract_features, load_gray
from search_drawings import similarity_score, list_drawings

SEED = 42  # 결과를 매번 같게 만들기 위한 값


# ---------------------------------------------------------
# 변형 함수들
# ---------------------------------------------------------
def rotate(gray, angle):
    """기울이기. 잘리지 않게 캔버스를 키우고, 빈 곳은 흰색으로 채운다."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        gray, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def shrink(gray, ratio):
    """크기 줄이기."""
    h, w = gray.shape
    return cv2.resize(
        gray, (max(1, int(w * ratio)), max(1, int(h * ratio))),
        interpolation=cv2.INTER_AREA,
    )


def add_noise(gray, rng, sigma=25, sp=0.01):
    """얼룩(가우시안)과 점(소금후추) 노이즈 추가."""
    noisy = gray.astype(np.float32) + rng.normal(0, sigma, gray.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    mask = rng.random(gray.shape)
    noisy[mask < sp / 2] = 0
    noisy[mask > 1 - sp / 2] = 255
    return noisy


CONDITIONS = {
    "원본 그대로": lambda g, rng: g,
    "회전 5도": lambda g, rng: rotate(g, 5),
    "회전 15도": lambda g, rng: rotate(g, 15),
    "50% 축소": lambda g, rng: shrink(g, 0.5),
    "노이즈": lambda g, rng: add_noise(g, rng),
    "복합(회전5+70%축소+노이즈)": lambda g, rng: add_noise(shrink(rotate(g, 5), 0.7), rng),
}


# ---------------------------------------------------------
# 평가
# ---------------------------------------------------------
def rank_of_original(scores, true_id, twins):
    """
    원본의 순위. 점수가 같은 경우는 불리하게(원본 뒤로) 센다.
    scores: {출원번호: 유사도}
    twins : 원본과 파일 내용이 완전히 같은 다른 도면들의 출원번호.
            같은 그림이므로 정답과 같은 취급을 하고, 순위 계산에서 뺀다.
    """
    true_score = scores[true_id]
    skip = set(twins) | {true_id}
    better_or_equal = sum(1 for pid, s in scores.items() if pid not in skip and s >= true_score)
    return 1 + better_or_equal


def find_identical_files(ids, paths):
    """파일 내용(해시)이 완전히 같은 도면끼리 묶어서 {출원번호: {같은 도면 출원번호들}} 반환."""
    groups = {}
    for pid, p in zip(ids, paths):
        with open(p, "rb") as f:
            groups.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(pid)
    twins = {pid: set() for pid in ids}
    for members in groups.values():
        if len(members) > 1:
            for pid in members:
                twins[pid] = set(members) - {pid}
    return twins


def main():
    base = os.path.dirname(__file__)
    img_dir = os.path.join(base, "..", "images")
    paths = list_drawings(img_dir)
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    print(
        f"검색 대상: 도면 {len(paths)}장 / 평가 쿼리: 조건별 {len(paths)}장 "
        f"(변형 조건 {len(CONDITIONS)}개, 총 {len(paths) * len(CONDITIONS)}건)"
    )

    # 내용이 완전히 같은 도면은 같은 정답으로 인정 (순위 계산에서 서로 제외)
    twins = find_identical_files(ids, paths)
    dup = sorted({tuple(sorted(s | {pid})) for pid, s in twins.items() if s})
    for group in dup:
        print(f"내용이 같은 도면(같은 정답으로 처리): {', '.join(group)}")
    print()

    # 1) 검색 대상(갤러리) 특징을 미리 한 번만 계산
    t0 = time.time()
    gallery = {}
    for pid, p in zip(ids, paths):
        _, desc, info = extract_features(p)
        gallery[pid] = (desc, info["num_keypoints"])
    print(f"검색 대상 특징 추출 완료 ({time.time() - t0:.0f}초)\n")

    # 2) 변형 조건별로 모든 도면을 쿼리로 검색
    ranks = {cond: {} for cond in CONDITIONS}  # ranks[조건][출원번호] = 원본 순위

    with tempfile.TemporaryDirectory() as tmp:
        for cond, fn in CONDITIONS.items():
            rng = np.random.default_rng(SEED)
            t1 = time.time()
            for pid, p in zip(ids, paths):
                gray = load_gray(p)
                distorted = fn(gray, rng)

                tmp_path = os.path.join(tmp, "q.png")  # PNG: 저장 과정에서 화질이 더 깎이지 않게
                cv2.imwrite(tmp_path, distorted)
                _, q_desc, q_info = extract_features(tmp_path)

                scores = {}
                for gid, (g_desc, g_n) in gallery.items():
                    score, _ = similarity_score(q_desc, q_info["num_keypoints"], g_desc, g_n)
                    scores[gid] = score
                ranks[cond][pid] = rank_of_original(scores, pid, twins[pid])

            r = list(ranks[cond].values())
            top1 = sum(1 for x in r if x == 1) / len(r)
            print(f"[{cond}] 완료 ({time.time() - t1:.0f}초)  Top-1 {top1 * 100:.0f}%")

    # 3) 결과 표
    n = len(ids)
    print("\n" + "=" * 70)
    print(f"{'변형 조건':28} | {'Top-1':>7} | {'Hit@5':>8} | {'순위 중앙값':>9}")
    print("-" * 70)
    for cond in CONDITIONS:
        r = list(ranks[cond].values())
        top1 = sum(1 for x in r if x == 1) / n
        hit5 = sum(1 for x in r if x <= 5) / n
        med = float(np.median(r))
        print(f"{cond:28} | {top1 * 100:>6.0f}% | {hit5 * 100:>7.0f}% | {med:>9.0f}")

    # 4) 원본을 1위로 못 찾은 경우 목록
    print("\n원본을 1위로 못 찾은 도면 (조건: 순위)")
    any_fail = False
    for pid in ids:
        fails = [f"{c}: {ranks[c][pid]}위" for c in CONDITIONS if ranks[c][pid] > 1]
        if fails:
            any_fail = True
            print(f"  {pid}  ->  " + ", ".join(fails))
    if not any_fail:
        print("  없음 (전부 1위)")

    # 5) CSV 저장
    out = os.path.join(base, "eval_results.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["출원번호"] + list(CONDITIONS))
        for pid in ids:
            w.writerow([pid] + [ranks[c][pid] for c in CONDITIONS])
    print(f"\n도면별 순위 저장: {out}")


if __name__ == "__main__":
    main()
