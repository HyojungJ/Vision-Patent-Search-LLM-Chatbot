"""
[진단] KIPRIS accessKey 유효성 + 특허 서지 엔드포인트 응답 원문 확인

1) 이미 검증된 IPC 엔드포인트(ClassificationService/searchIPCInfo)를 이 키로 호출
   -> 키 자체가 유효한지 확정
2) 특허 서지 후보 엔드포인트들의 '응답 원문 앞부분'을 그대로 출력
   -> 200인데 HTML이 오는 이유(에러 메시지/필요 파라미터) 파악
"""

import os
import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH)
ACCESS_KEY = os.getenv("KIPRIS_ACCESS_KEY", "").strip()

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
TEST_APP_NO = "1020050108060"
BASE = "http://plus.kipris.or.kr/openapi/rest"


def show(title, url, params):
    print("\n" + "=" * 70)
    print(f"[{title}]")
    print(f"URL   : {url}")
    print(f"params: { {k: (v[:6]+'...' if k=='accessKey' else v) for k,v in params.items()} }")
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        print(f"status: {r.status_code}, content-type: {r.headers.get('Content-Type','')}")
        body = (r.text or "")
        # 원문 앞부분 출력 (줄바꿈 정리)
        preview = " ".join(body.split())[:600]
        print(f"body preview:\n{preview}")
    except Exception as e:
        print(f"EXC: {e}")


def main():
    if not ACCESS_KEY or ACCESS_KEY.startswith("여기에"):
        print("[ERROR] KIPRIS_ACCESS_KEY 미설정")
        return

    print(f"ACCESS_KEY 길이 {len(ACCESS_KEY)}, 앞 4자리 {ACCESS_KEY[:4]}...")

    # 1) 검증된 IPC 엔드포인트로 키 유효성 확인
    show(
        "키 유효성 확인 (IPC 서비스)",
        f"{BASE}/ClassificationService/searchIPCInfo",
        {"ipcNumber": "G06T 7/00", "accessKey": ACCESS_KEY},
    )

    # 2) 특허 서지 후보들의 원문 확인
    for name, ep, param in [
        ("서지상세 후보 A", "patUtiliInfoSearchSevice/getBibliographyDetailInfoSearch", "applicationNumber"),
        ("서지상세 후보 B", "ScreeningService/applicationNumberSearchInfo", "applicationNumber"),
        ("서지상세 후보 C", "patUtiliInfoSearchSevice/getBibliographySumryInfoSearch", "applicationNumber"),
    ]:
        show(name, f"{BASE}/{ep}", {param: TEST_APP_NO, "accessKey": ACCESS_KEY})


if __name__ == "__main__":
    main()
