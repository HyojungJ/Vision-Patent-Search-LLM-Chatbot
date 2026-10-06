"""
[특허 도면 특징 추출 모듈 - OpenCV]

도면 이미지 1장에서 '구조적 특징'을 뽑는다. 딥러닝 없이 OpenCV만 사용한다.

파이프라인:
  1. 그레이스케일 변환
  2. 노이즈 제거 (median blur) + 대비 보정
  3. (크기 정규화) 도면마다 해상도가 달라 특징점 수가 들쭉날쭉하므로 긴 변 기준 리사이즈
  4. ORB 특징점 + 디스크립터 추출

왜 ORB인가:
  - 특허 도면은 '흰 배경 + 검은 선'이라 코너/엣지 특징점이 뚜렷하다.
  - ORB는 회전/스케일에 어느 정도 강인하고, 디스크립터가 이진(binary)이라
    해밍 거리로 빠르게 매칭할 수 있다.
  - SIFT와 달리 라이선스/속도 부담이 적어 포트폴리오 데모에 적합하다.

이 모듈은 유사 도면 검색(search_drawings.py)의 공통 토대로 사용된다.
"""

import os

# cv2가 다른 OpenMP 런타임과 충돌하는 환경 대비
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

# 특징점 최대 개수 (너무 많으면 매칭이 느려지고 노이즈도 섞임)
ORB_N_FEATURES = 1000
# 긴 변을 이 크기로 맞춰 도면 간 스케일 편차를 줄인다
NORMALIZE_LONG_SIDE = 1000


def load_gray(image_path: str) -> np.ndarray:
    """이미지를 그레이스케일로 읽는다. 실패 시 예외."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return img


def preprocess(gray: np.ndarray) -> np.ndarray:
    """
    도면 전처리:
      - median blur로 점 노이즈 제거(선은 보존)
      - CLAHE로 국소 대비를 높여 흐린 선도 또렷하게
      - 긴 변 기준 리사이즈로 스케일 정규화
    """
    # 점 노이즈 제거
    den = cv2.medianBlur(gray, 3)

    # 국소 대비 향상 (스캔 품질이 낮은 도면의 흐린 선 보강)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(den)

    # 스케일 정규화 (긴 변을 NORMALIZE_LONG_SIDE로)
    h, w = enhanced.shape
    long_side = max(h, w)
    if long_side != NORMALIZE_LONG_SIDE:
        scale = NORMALIZE_LONG_SIDE / long_side
        enhanced = cv2.resize(
            enhanced, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    return enhanced


def extract_features(image_path: str):
    """
    이미지 경로 -> (keypoints, descriptors, debug_info)
      - keypoints: cv2.KeyPoint 리스트
      - descriptors: np.ndarray(shape=(N,32), uint8) 또는 None(특징점 없음)
      - debug_info: dict(num_keypoints, size)
    """
    gray = load_gray(image_path)
    proc = preprocess(gray)

    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(proc, None)

    info = {
        "num_keypoints": 0 if keypoints is None else len(keypoints),
        "size": proc.shape[::-1],  # (w, h)
    }
    return keypoints, descriptors, info


if __name__ == "__main__":
    # 자가 점검: images 폴더의 각 도면에서 특징점 몇 개가 잡히는지 출력
    base = os.path.dirname(__file__)
    img_dir = os.path.join(base, "..", "images")
    if not os.path.isdir(img_dir):
        print(f"[안내] 이미지 폴더가 없습니다: {os.path.abspath(img_dir)}")
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
        print(f"이미지 {len(files)}장 특징점 추출 결과:\n")
        print(f"{'파일명':28} | {'특징점수':>7} | 크기")
        print("-" * 55)
        for name in files:
            path = os.path.join(img_dir, name)
            try:
                _, _, info = extract_features(path)
                print(f"{name:28} | {info['num_keypoints']:>7} | {info['size']}")
            except Exception as e:
                print(f"{name:28} | ERROR: {e}")
