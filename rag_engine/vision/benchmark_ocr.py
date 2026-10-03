"""
[전처리 가치 벤치마크]
질문: OpenCV 전처리(이진화/디스큐/노이즈제거)가 EasyOCR 성능을 실제로 올리는가?

방법:
  노이즈 강도를 여러 단계로 올려가며(clean -> 심한 열화),
  각 단계에서 '전처리 X' vs '전처리 O'의 참조번호 recall을 비교한다.

목적: 전처리를 파이프라인에 넣을지, 넣는다면 어떤 조건에서 켤지 데이터로 결정.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

from make_test_drawing import make_clean_drawing, GROUND_TRUTH_LABELS
from reference_ocr import detect_reference_numbers, evaluate


def degrade(img, angle, gauss_sigma, sp_ratio, blur_k):
    """다양한 강도의 열화를 적용."""
    h, w = img.shape
    if abs(angle) > 0.01:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if gauss_sigma > 0:
        noise = np.random.normal(0, gauss_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if sp_ratio > 0:
        n = int(sp_ratio * img.size)
        ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n); img[ys, xs] = 0
        ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n); img[ys, xs] = 255
    if blur_k >= 3:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    return img


LEVELS = [
    # 이름,          angle, gauss, s&p,   blur
    ("clean",         0.0,   0,     0.0,   0),
    ("light",         2.0,   12,    0.002, 3),
    ("medium",        3.5,   22,    0.006, 3),
    ("heavy",         5.0,   35,    0.012, 5),
    ("severe",        7.0,   45,    0.020, 5),
]


def main():
    clean = make_clean_drawing()
    gt = GROUND_TRUTH_LABELS

    print(f"정답 {len(gt)}개: {gt}\n")
    header = f"{'level':10} | {'전처리X recall':>14} | {'전처리O recall':>14} | 승자"
    print(header)
    print("-" * len(header))

    for name, angle, gauss, sp, blur in LEVELS:
        np.random.seed(7)
        img = degrade(clean.copy(), angle, gauss, sp, blur)

        det_raw = detect_reference_numbers(img, use_preprocess=False)
        det_pp = detect_reference_numbers(img, use_preprocess=True)
        r_raw = evaluate(det_raw, gt)["recall"]
        r_pp = evaluate(det_pp, gt)["recall"]

        if r_pp > r_raw:
            winner = "전처리 O"
        elif r_raw > r_pp:
            winner = "전처리 X"
        else:
            winner = "동점"
        print(f"{name:10} | {r_raw:>14.3f} | {r_pp:>14.3f} | {winner}")


if __name__ == "__main__":
    main()
