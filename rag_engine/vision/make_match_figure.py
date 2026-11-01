"""
[시각자료] Lowe's ratio test 적용 전 / 후 특징점 매칭 비교 그림 생성

두 도면(쿼리 vs 유사 도면)의 ORB 특징점을 매칭해서,
- 위: ratio test 적용 전 (모든 매칭, 오매칭 포함 -> 선이 지저분)
- 아래: ratio test 적용 후 (확실한 매칭만 -> 깔끔)
를 한 장의 이미지로 저장한다.

결과: match_figure.png  (PPT에 바로 사용)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

from drawing_features import preprocess, load_gray, ORB_N_FEATURES

RATIO = 0.75  # search_drawings.py와 동일한 기준


def kp_desc(path):
    """전처리 후 ORB 특징점/디스크립터 + 전처리된 이미지 반환."""
    proc = preprocess(load_gray(path))
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    kp, desc = orb.detectAndCompute(proc, None)
    return proc, kp, desc


def draw_banner(img, text):
    """이미지 위에 설명 띠(배너)를 올린다."""
    h, w = img.shape[:2]
    banner_h = 46
    canvas = np.full((h + banner_h, w, 3), 255, np.uint8)
    canvas[banner_h:] = img
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (40, 40, 40), -1)
    cv2.putText(canvas, text, (14, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    base = os.path.dirname(__file__)
    query = os.path.join(base, "..", "..", "test.jpg")
    cand = os.path.join(base, "..", "images", "1020160183317.jpg")

    img1, kp1, d1 = kp_desc(query)
    img2, kp2, d2 = kp_desc(cand)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)

    # --- ratio test 적용 전: 각 쌍의 1등 매칭을 '전부' 사용 (필터 없음) ---
    all_matches = [pair[0] for pair in knn if len(pair) >= 1]

    # --- ratio test 적용 후: 그중 확실한 매칭만 남김 ---
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < RATIO * n.distance]

    # BEFORE가 AFTER보다 항상 많아야 효과가 바르게 보임 (전체 >= 걸러진 것)
    print(f"적용 전 전체 매칭: {len(all_matches)}개")
    print(f"적용 후 좋은 매칭: {len(good)}개  (전체 중 {len(good)/max(1,len(all_matches))*100:.0f}%만 통과)")

    # 매칭 그림 그리기
    before = cv2.drawMatches(
        img1, kp1, img2, kp2, all_matches, None,
        matchColor=(0, 0, 255), singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    after = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        matchColor=(0, 170, 0), singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    before = draw_banner(before, f"BEFORE ratio test : {len(all_matches)} matches (noisy, wrong matches included)")
    after = draw_banner(after, f"AFTER ratio test : {len(good)} reliable matches only")

    # 두 폭을 맞춰 세로로 결합
    w = max(before.shape[1], after.shape[1])
    def pad(im):
        if im.shape[1] < w:
            im = cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1],
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return im
    gap = np.full((20, w, 3), 255, np.uint8)
    combined = np.vstack([pad(before), gap, pad(after)])

    out = os.path.join(base, "match_figure.png")
    cv2.imwrite(out, combined)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
