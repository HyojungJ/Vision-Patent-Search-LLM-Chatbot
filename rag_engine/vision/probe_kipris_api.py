"""
[2단계 - 엔드포인트 탐색 + 신선한 도면 URL 확보 검증]

목적:
  KIPRIS Plus OpenAPI(accessKey 방식)로 특허 서지상세 서비스를 호출해서
  1) 어떤 엔드포인트가 우리가 가진 출원번호에 대해 정상 응답(resultCode 00)을 주는지 찾고
  2) 그 응답 안에 <imagePathInfo>(대표도면 URL)가 들어있는지 확인하고
  3) 그 '신선한' largePath URL로 실제 이미지가 다운로드되는지까지 검증한다.

이 스크립트가 성공하면 -> 도면 수집 파이프라인 구축이 확정된다.

사용법:
  1) rag_engine/.env 의 KIPRIS_ACCESS_KEY 에 발급받은 키를 넣는다.
  2) python probe_kipris_api.py
"""

import os
import re
import sys

import requests
from dotenv import load_dotenv

# rag_engine/.env 로드 (이 파일은 rag_engine/vision/ 에 있으므로 한 단계 위)
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH)

ACCESS_KEY = os.getenv("KIPRIS_ACCESS_KEY", "").strip()

# 검증에 사용할 출원번호 (우리 데이터에 실제 존재하는 것)
# API는 보통 하이픈 없는 13자리를 그대로 받는다.
TEST_APP_NO = "1020050108060"

# 특허 서지/상세 서비스의 알려진 엔드포인트 후보들.
# 실제로 두드려 보고 어떤 것이 imagePathInfo 를 주는지 자동으로 판별한다.
BASE = "http://plus.kipris.or.kr/openapi/rest"
ENDPOINT_CANDIDATES = [
    # (엔드포인트 URL, 출원번호 파라미터 이름)
    (f"{BASE}/patUtiliInfoSearchSevice/patentUtilityDetailInfo", "applicationNumber"),
    (f"{BASE}/patUtiliInfoSearchSevice/getBibliographyDetailInfoSearch", "applicationNumber"),
    (f"{BASE}/patUtiliInfoSearchSevice/getBiblioSummaryInfoSearch", "applicationNumber"),
    (f"{BASE}/PatentUtilityService/patentUtilityDetailInfo", "applicationNumber"),
    (f"{BASE}/ScreeningService/applicationNumberSearchInfo", "applicationNumber"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
}


def extract_first(pattern, text):
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def probe_endpoint(url, appno_param):
    """단일 엔드포인트를 호출하고 결과를 딕셔너리로 요약."""
    params = {appno_param: TEST_APP_NO, "accessKey": ACCESS_KEY}
    info = {"url": url, "ok": False, "status": None, "resultCode": "",
            "resultMsg": "", "has_image": False, "image_url": "", "note": ""}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        info["status"] = r.status_code
        text = r.text or ""

        info["resultCode"] = extract_first(r"<resultCode>(.*?)</resultCode>", text)
        info["resultMsg"] = extract_first(r"<resultMsg>(.*?)</resultMsg>", text)

        # 이미지 경로 탐색
        large = extract_first(r"<largePath>(.*?)</largePath>", text)
        path = extract_first(r"<path>(.*?)</path>", text)
        image_url = large or path
        if image_url:
            info["has_image"] = True
            info["image_url"] = image_url

        # 성공 판정: HTTP 200 + resultCode가 00 계열
        if r.status_code == 200 and info["resultCode"].startswith("00"):
            info["ok"] = True
        elif r.status_code == 200 and not info["resultCode"]:
            # resultCode 태그 자체가 없으면 응답 앞부분을 노트로 남김
            info["note"] = text[:150].replace("\n", " ")

    except Exception as e:
        info["note"] = f"EXC: {e}"
    return info


def download_image(image_url, out_path):
    r = requests.get(image_url, headers=HEADERS, timeout=30, allow_redirects=True)
    ctype = r.headers.get("Content-Type", "").lower()
    is_img = ctype.startswith("image") or (
        len(r.content) > 1000 and r.content[:3] == b"\xff\xd8\xff"
    )
    if r.status_code == 200 and is_img:
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True, len(r.content), ctype
    return False, len(r.content), ctype


def main():
    if not ACCESS_KEY or ACCESS_KEY.startswith("여기에"):
        print("[ERROR] KIPRIS_ACCESS_KEY 가 .env 에 설정되지 않았습니다.")
        print(f"        확인 경로: {os.path.abspath(ENV_PATH)}")
        sys.exit(1)

    print(f"ACCESS_KEY 로드됨 (길이 {len(ACCESS_KEY)}, 앞 4자리 {ACCESS_KEY[:4]}...)")
    print(f"테스트 출원번호: {TEST_APP_NO}\n")
    print("=" * 70)

    winner = None
    for url, appno_param in ENDPOINT_CANDIDATES:
        print(f"\n[probe] {url}")
        info = probe_endpoint(url, appno_param)
        print(f"  status={info['status']} resultCode='{info['resultCode']}' "
              f"resultMsg='{info['resultMsg']}'")
        print(f"  has_image={info['has_image']}")
        if info["note"]:
            print(f"  note: {info['note']}")
        if info["ok"] and info["has_image"]:
            print("  ✅ 이 엔드포인트가 도면 URL을 반환합니다.")
            winner = info
            break
        elif info["ok"]:
            print("  △ 정상 응답이지만 imagePathInfo 없음 (다른 엔드포인트일 수 있음)")

    print("\n" + "=" * 70)
    if not winner:
        print("도면 URL을 주는 엔드포인트를 아직 찾지 못했습니다.")
        print("=> 위 로그의 resultMsg를 확인하세요. 흔한 원인:")
        print("   - 해당 서비스가 아직 승인 대기중(신청 직후)")
        print("   - accessKey에 이 서비스가 포함되지 않음")
        print("   - 엔드포인트 이름이 위 후보와 다름 (KIPRIS 마이페이지의 샘플 URL 필요)")
        sys.exit(0)

    # 신선한 URL로 실제 이미지 다운로드 검증
    print(f"신선한 도면 URL 확보:\n  {winner['image_url'][:90]}...")
    out_dir = os.path.join(os.path.dirname(__file__), "sample_downloads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{TEST_APP_NO}_fresh.jpg")

    ok, nbytes, ctype = download_image(winner["image_url"], out_path)
    if ok:
        print(f"  ✅ 이미지 다운로드 성공: {out_path} ({nbytes} bytes, {ctype})")
        print("\n=> 2단계 검증 완료. 도면 수집 파이프라인 구축 가능.")
        print(f"   승인된 엔드포인트: {winner['url']}")
    else:
        print(f"  ❌ URL은 받았으나 이미지 다운로드 실패 (bytes={nbytes}, ctype={ctype})")


if __name__ == "__main__":
    main()
