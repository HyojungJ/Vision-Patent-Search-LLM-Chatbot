"""
[테스트 도면 생성]
KIPRIS 승인 대기 중, 비전 파이프라인을 검증하기 위한 '특허 도면 스타일' 테스트 이미지를 생성한다.

특허 도면의 핵심 특징을 재현한다:
  - 흰 배경 + 검은 선(line drawing)
  - 부품 도형 (사각형, 원)
  - 참조번호 라벨 (101, 102, ... ) + 인출선(leader line)
  - "도 1 (FIG. 1)" 같은 도면 제목

또한 '저품질 스캔' 버전도 만든다:
  - 회전(기울어짐), 가우시안 노이즈, salt&pepper 노이즈, 흐림

정답(ground truth) 참조번호를 알고 있으므로, OCR 인식률을 정량 측정할 수 있다.
KIPRIS 승인 후에는 실제 도면으로 동일 파이프라인을 검증한다.
"""

import os
import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_drawings")

# 정답 참조번호 (검증용)
GROUND_TRUTH_LABELS = ["100", "101", "102", "110", "120", "121", "130", "200"]


def make_clean_drawing(width=1000, height=750):
    """깨끗한 특허 도면 스타일 이미지 생성."""
    img = np.full((height, width), 255, dtype=np.uint8)  # 흰 배경

    def line(p1, p2, t=2):
        cv2.line(img, p1, p2, 0, t)

    def rect(x, y, w, h, t=2):
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, t)

    def circle(c, r, t=2):
        cv2.circle(img, c, r, 0, t)

    def label(text, org):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)

    # 도면 제목
    cv2.putText(img, "FIG. 1", (width // 2 - 60, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)

    # 외곽 하우징 (100)
    rect(120, 110, 760, 540)
    label("100", (95, 100))
    line((150, 108), (110, 92))  # leader

    # 내부 블록 1 (110)
    rect(180, 190, 260, 180)
    label("110", (300, 175))
    line((310, 190), (330, 172))

    # 내부 블록 2 (120) + 세부 (121)
    rect(520, 190, 280, 180)
    label("120", (620, 172))
    line((660, 200), (648, 180))       # 인출선은 라벨과 간격을 둔다
    circle((660, 285), 45)
    label("121", (735, 292))
    line((708, 288), (725, 288))       # 라벨에서 살짝 떨어뜨림

    # 하단 블록 (130)
    rect(300, 445, 400, 150)
    label("130", (470, 428))
    line((505, 445), (492, 432))

    # 연결선
    line((310, 370), (310, 445))
    line((660, 370), (660, 445))
    line((440, 275), (520, 275))
    label("101", (452, 262))
    label("102", (452, 322))

    # 별도 부품 (200) - 오른쪽 끝, 인출선을 왼쪽으로 빼서 라벨과 분리
    circle((880, 400), 55, 2)
    label("200", (948, 408))
    line((925, 400), (938, 400))       # 원과 라벨 사이 짧은 인출선

    return img


def degrade(img, angle=3.5, noise_sigma=18):
    """저품질 스캔 효과: 회전 + 가우시안/솔트페퍼 노이즈 + 약한 블러."""
    h, w = img.shape
    # 회전 (기울어진 스캔)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 가우시안 노이즈
    noise = np.random.normal(0, noise_sigma, rotated.shape).astype(np.float32)
    noisy = np.clip(rotated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # salt & pepper
    sp = noisy.copy()
    n = int(0.004 * sp.size)
    ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n)
    sp[ys, xs] = 0
    ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n)
    sp[ys, xs] = 255

    # 약한 블러
    blurred = cv2.GaussianBlur(sp, (3, 3), 0)
    return blurred


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    clean = make_clean_drawing()
    clean_path = os.path.join(OUT_DIR, "patent_fig_clean.png")
    cv2.imwrite(clean_path, clean)
    print(f"생성: {clean_path}")

    np.random.seed(42)
    noisy = degrade(clean)
    noisy_path = os.path.join(OUT_DIR, "patent_fig_scanned.png")
    cv2.imwrite(noisy_path, noisy)
    print(f"생성: {noisy_path}")

    print(f"\n정답 참조번호 ({len(GROUND_TRUTH_LABELS)}개): {GROUND_TRUTH_LABELS}")


if __name__ == "__main__":
    main()
