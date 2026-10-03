"""
[OCR 디버그] 검출 결과를 원본 이미지 위에 그려서 어디를 어떻게 읽었는지 눈으로 확인.
또한 allowlist 없이(원문 그대로) 읽은 결과도 같이 출력해 오인식 패턴을 파악.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
from preprocess import preprocess_for_ocr
from reference_ocr import get_reader, _to_rgb


def visualize(name, use_pp):
    base = os.path.dirname(__file__)
    path = os.path.join(base, "test_drawings", name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if use_pp:
        proc = preprocess_for_ocr(img)
    else:
        proc = img
    rgb = _to_rgb(proc)

    reader = get_reader()
    # allowlist 없이 원문 그대로 읽기 (오인식 패턴 관찰용)
    raw = reader.readtext(rgb)

    vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR) if proc.ndim == 2 else proc.copy()
    print(f"\n=== {name} (전처리={'O' if use_pp else 'X'}) 원문 OCR 결과 ===")
    for bbox, text, conf in raw:
        pts = np.array(bbox, dtype=int)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        x, y = pts[0]
        cv2.putText(vis, f"{text}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"  '{text}'  conf={conf:.2f}  pos=({pts[:,0].mean():.0f},{pts[:,1].mean():.0f})")

    tag = "pp" if use_pp else "raw"
    out = os.path.join(base, "preprocess_output", f"ocr_debug_{os.path.splitext(name)[0]}_{tag}.png")
    cv2.imwrite(out, vis)
    print(f"  -> 시각화 저장: {out}")


if __name__ == "__main__":
    for name in ["patent_fig_clean.png", "patent_fig_scanned.png"]:
        visualize(name, use_pp=False)
        visualize(name, use_pp=True)
