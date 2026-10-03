"""
[특허 도면 전처리 모듈 - OpenCV]

특허 도면(선화, line drawing)의 스캔 품질을 개선해 OCR/검색에 적합한 형태로 만든다.

파이프라인:
  1. 그레이스케일 변환
  2. 노이즈 제거 (median blur로 salt&pepper 제거 + bilateral로 엣지 보존)
  3. Otsu 이진화 (적응형 임계값)
  4. 디스큐 (기울어진 스캔 각도 자동 검출 후 회전 보정)

설계 의도:
  - 특허 도면은 '흰 배경 + 검은 선'이라 이진화가 잘 먹힌다.
  - 얇은 선/작은 숫자를 보존해야 하므로, 과한 blur/erosion은 피한다.
  - 디스큐는 참조번호 OCR 정확도에 직접 영향을 주므로 중요하다.

이 모듈은 KIPRIS 승인 후 실제 도면에도 그대로 적용된다.
"""

import os

# cv2 + torch(easyocr) 동시 사용 시 OpenMP 런타임 중복 충돌 회피
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """컬러/그레이 무엇이 들어와도 단일 채널 그레이스케일로 통일."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(gray: np.ndarray) -> np.ndarray:
    """
    노이즈 제거.
    - median blur: salt&pepper(점 노이즈) 제거에 효과적, 엣지 보존.
    - bilateral filter: 가우시안 노이즈를 줄이되 선(엣지)은 살림.
    """
    med = cv2.medianBlur(gray, 3)
    bil = cv2.bilateralFilter(med, d=5, sigmaColor=50, sigmaSpace=50)
    return bil


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Otsu 이진화. 반환은 '흰 배경(255) + 검은 선(0)' 형태를 유지.
    THRESH_BINARY + Otsu로 자동 임계값 결정.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def estimate_skew_angle(binary: np.ndarray) -> float:
    """
    기울기 각도(도 단위) 추정.
    검은 선 픽셀들의 최소 외접 사각형 각도를 이용한다.
    특허 도면은 수평/수직 선이 많아 이 방법이 안정적이다.
    반환: 이미지를 몇 도 회전시켜야 수평이 맞는지(양수=반시계).
    """
    # 검은 선(0) 픽셀 좌표 (binary: 배경 255, 선 0)
    inv = cv2.bitwise_not(binary)  # 선이 255가 됨
    coords = np.column_stack(np.where(inv > 0))
    if coords.shape[0] < 50:
        return 0.0

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))  # (x,y) 순서로
    angle = rect[-1]

    # minAreaRect의 각도는 [-90,0) 범위 -> 실사용 각도로 정규화
    if angle < -45:
        angle = 90 + angle
    # 미세한 각도만 보정 대상으로 (오검출 방지)
    if abs(angle) > 15:
        return 0.0
    return -angle  # warpAffine에 넘길 회전각


def deskew(gray: np.ndarray, angle: float, linear: bool = True) -> np.ndarray:
    """
    추정된 각도만큼 회전 보정. 배경은 흰색(255)으로 채움.
    linear=True면 LINEAR 보간(그레이스케일 단계에 적합), False면 NEAREST(이진 보존).
    """
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess(img: np.ndarray, do_deskew: bool = True):
    """
    전체 전처리 파이프라인 실행.

    순서가 중요하다:
      grayscale -> (기울기 추정용 임시 이진화) -> grayscale 단계에서 deskew
      -> denoise -> 최종 binarize
    즉 '이진화된 이미지를 회전'시키지 않는다. 이진 이미지를 보간 회전하면
    획이 뭉개져 OCR 정확도가 떨어지기 때문. 회전은 그레이스케일에서 수행한다.

    반환: dict(단계별 이미지) - 디버깅/시각 비교용.
      keys: gray, deskewed_gray, denoised, binary, deskewed, angle
    """
    gray = to_grayscale(img)

    angle = 0.0
    deskewed_gray = gray
    if do_deskew:
        # 각도 추정은 이진 이미지가 안정적이므로 임시 이진화로 각도만 구한다
        tmp_bin = binarize(denoise(gray))
        angle = estimate_skew_angle(tmp_bin)
        # 실제 회전은 그레이스케일에 LINEAR 보간으로 (획 보존)
        deskewed_gray = deskew(gray, angle, linear=True)

    denoised = denoise(deskewed_gray)
    binary = binarize(denoised)

    return {
        "gray": gray,
        "deskewed_gray": deskewed_gray,
        "denoised": denoised,
        "binary": binary,
        "deskewed": binary,   # 하위호환: OCR 입력으로 쓰는 최종 이진 이미지
        "angle": angle,
    }


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """OCR 입력용 최종 이미지(디스큐 후 이진화된 이미지)만 반환."""
    return preprocess(img, do_deskew=True)["binary"]


# ---------------------------------------------------------
# 단독 실행: 테스트 도면에 전처리를 적용하고 단계별 결과 저장
# ---------------------------------------------------------
def _demo():
    base = os.path.dirname(__file__)
    in_dir = os.path.join(base, "test_drawings")
    out_dir = os.path.join(base, "preprocess_output")
    os.makedirs(out_dir, exist_ok=True)

    for name in ["patent_fig_clean.png", "patent_fig_scanned.png"]:
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            print(f"[SKIP] 파일 없음: {path}")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        stem = os.path.splitext(name)[0]
        result = preprocess(img)

        print(f"\n[{name}] 추정 기울기 보정각 = {result['angle']:.2f}도")
        for step in ["denoised", "binary", "deskewed"]:
            out_path = os.path.join(out_dir, f"{stem}__{step}.png")
            cv2.imwrite(out_path, result[step])
            print(f"  저장: {out_path}")


if __name__ == "__main__":
    _demo()
