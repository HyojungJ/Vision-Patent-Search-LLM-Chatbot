"""
[참조번호 OCR 모듈]

특허 도면 안의 '부품 참조번호'(101, 102, 110 ...)를 인식한다.
이 번호들은 명세서 텍스트의 "101: 하우징" 같은 설명과 연결되므로,
도면-텍스트를 잇는 핵심 고리다.

엔진: EasyOCR (pip만으로 설치 완결, 이미 torch 존재).
전략:
  - preprocess.py의 전처리를 거친 이미지를 OCR에 투입.
  - 특허 참조번호는 대부분 '숫자'이므로 allowlist='0123456789'로 정밀도를 높인다.
  - 결과에서 순수 숫자 토큰만 남기고, 위치(bbox 중심)와 신뢰도를 함께 반환.

주의: EasyOCR Reader 초기화는 무겁다(모델 로드). 모듈 전역에서 1회만 생성.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import cv2
import numpy as np

from preprocess import preprocess_for_ocr

# EasyOCR Reader는 최초 1회만 생성 (지연 초기화)
_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        # 특허 도면 라벨은 숫자 위주지만, 'FIG' 등 영문도 있으므로 en 로드
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """EasyOCR은 3채널을 기대. 단일 채널이면 3채널로 확장."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_reference_numbers(
    img: np.ndarray,
    use_preprocess: bool = True,
    min_confidence: float = 0.3,
    numeric_only: bool = True,
):
    """
    도면 이미지에서 참조번호를 검출한다.

    Parameters
    ----------
    img : np.ndarray  입력 이미지(그레이/컬러)
    use_preprocess : bool  True면 preprocess_for_ocr 적용 후 OCR
    min_confidence : float  이 신뢰도 미만은 버림
    numeric_only : bool  True면 순수 숫자 토큰만 남김

    Returns
    -------
    list[dict] : 각 항목 { "text", "confidence", "center": (x,y), "bbox": [...] }
    """
    if use_preprocess:
        proc = preprocess_for_ocr(img)
        ocr_input = _to_rgb(proc)
    else:
        ocr_input = _to_rgb(img)

    reader = get_reader()
    # allowlist를 강제하면 EasyOCR이 노이즈 영역을 억지로 숫자로 오인식(유령 숫자)한다.
    # 따라서 자유롭게 읽게 하고, 순수 숫자 토큰만 후처리로 남긴다.
    raw = reader.readtext(ocr_input)

    results = []
    for bbox, text, conf in raw:
        token = text.strip()
        if conf < min_confidence:
            continue
        if numeric_only:
            # 흔한 오인식 보정: O/o -> 0, 앞뒤 대시/공백 제거
            token = token.replace("O", "0").replace("o", "0")
            token = token.strip(" -_.")
            # '순수 숫자'만 참조번호로 인정 (예: 'FIG.1' 같은 혼합 토큰은 제외)
            if not re.fullmatch(r"\d{1,4}", token):
                continue
        # bbox: 4점 [[x,y],...] -> 중심 계산
        pts = np.array(bbox, dtype=np.float32)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        results.append({
            "text": token,
            "confidence": round(float(conf), 3),
            "center": (round(float(cx), 1), round(float(cy), 1)),
            "bbox": pts.astype(int).tolist(),
        })
    return results


def evaluate(detected, ground_truth):
    """
    검출 결과를 정답과 비교해 precision/recall 계산.
    detected: detect_reference_numbers 결과
    ground_truth: 정답 번호 리스트(문자열)
    """
    detected_set = set(d["text"] for d in detected)
    gt_set = set(ground_truth)

    tp = detected_set & gt_set
    fp = detected_set - gt_set
    fn = gt_set - detected_set

    precision = len(tp) / len(detected_set) if detected_set else 0.0
    recall = len(tp) / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "matched": sorted(tp),
        "false_positives": sorted(fp),
        "missed": sorted(fn),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ---------------------------------------------------------
# 단독 실행: 테스트 도면으로 OCR 검증 (전처리 유/무 비교)
# ---------------------------------------------------------
def _demo():
    base = os.path.dirname(__file__)
    in_dir = os.path.join(base, "test_drawings")
    ground_truth = ["100", "101", "102", "110", "120", "121", "130", "200"]

    for name in ["patent_fig_clean.png", "patent_fig_scanned.png"]:
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            print(f"[SKIP] {path}")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("\n" + "=" * 64)
        print(f"이미지: {name}")
        print(f"정답: {ground_truth}")

        for use_pp in [False, True]:
            tag = "전처리 O" if use_pp else "전처리 X"
            det = detect_reference_numbers(img, use_preprocess=use_pp)
            found = sorted(d["text"] for d in det)
            ev = evaluate(det, ground_truth)
            print(f"\n  [{tag}] 검출 {len(found)}개: {found}")
            print(f"    matched={ev['matched']}")
            print(f"    missed={ev['missed']} false_pos={ev['false_positives']}")
            print(f"    precision={ev['precision']} recall={ev['recall']} f1={ev['f1']}")


if __name__ == "__main__":
    _demo()
