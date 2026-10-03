"""
[1단계 검증 스크립트]
목적: 특허 XML의 <imagePathInfo> 안에 있는 도면 이미지 URL(largePath/path)로
      실제 JPG가 다운로드되는지 확인한다.

이 스크립트가 성공하면 -> 도면 다운로드 파이프라인 구축이 현실적이다.
실패하면(인증/세션 필요 등) -> 데이터 확보 전략을 먼저 재논의해야 한다.

사용법:
    python verify_drawing_download.py
"""

import os
import re
import sys
import glob

import requests

RAW_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "doc", "data", "raw_data"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_downloads")

# 검증용으로 몇 건만 시도
SAMPLE_COUNT = 5

IMG_BLOCK_RE = re.compile(r"<imagePathInfo>(.*?)</imagePathInfo>", re.DOTALL)
DOCNAME_RE = re.compile(r"<docName>(.*?)</docName>", re.DOTALL)
LARGEPATH_RE = re.compile(r"<largePath>(.*?)</largePath>", re.DOTALL)
PATH_RE = re.compile(r"<path>(.*?)</path>", re.DOTALL)


def extract_image_info(xml_text: str):
    """XML 문자열에서 첫 번째 imagePathInfo의 docName / largePath / path 추출."""
    block_m = IMG_BLOCK_RE.search(xml_text)
    if not block_m:
        return None
    block = block_m.group(1)

    def _first(rx):
        m = rx.search(block)
        return m.group(1).strip() if m else ""

    return {
        "docName": _first(DOCNAME_RE),
        "largePath": _first(LARGEPATH_RE),
        "path": _first(PATH_RE),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.xml")))
    if not xml_files:
        print(f"[ERROR] XML 파일을 찾지 못했습니다: {RAW_DATA_DIR}")
        sys.exit(1)

    print(f"총 XML 파일: {len(xml_files)}건 중 {SAMPLE_COUNT}건 검증 시도\n")

    headers = {
        # 브라우저처럼 보이게 하는 최소한의 헤더 (일부 서버가 UA 없는 요청 차단)
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "http://plus.kipris.or.kr/",
    }

    success = 0
    tried = 0

    for xml_path in xml_files:
        if tried >= SAMPLE_COUNT:
            break

        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        info = extract_image_info(xml_text)
        if not info or not (info["largePath"] or info["path"]):
            print(f"[SKIP] 이미지 경로 없음: {os.path.basename(xml_path)}")
            continue

        tried += 1
        patent_id = os.path.splitext(os.path.basename(xml_path))[0]
        url = info["largePath"] or info["path"]

        print(f"[{tried}] {patent_id}")
        print(f"    docName : {info['docName']}")
        print(f"    URL     : {url[:90]}...")

        try:
            resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            ctype = resp.headers.get("Content-Type", "")
            clen = len(resp.content)
            print(f"    -> status={resp.status_code}, "
                  f"content-type={ctype}, bytes={clen}")

            # 이미지인지 판단
            is_image = ctype.lower().startswith("image") or (
                clen > 1000 and resp.content[:3] == b"\xff\xd8\xff"  # JPEG magic
            )

            if resp.status_code == 200 and is_image:
                ext = ".jpg"
                out_path = os.path.join(OUT_DIR, f"{patent_id}{ext}")
                with open(out_path, "wb") as imgf:
                    imgf.write(resp.content)
                print(f"    ✅ 이미지 저장: {out_path}")
                success += 1
            else:
                # 실패 원인 파악을 위해 앞부분 미리보기
                preview = resp.content[:200]
                try:
                    preview_str = preview.decode("utf-8", errors="replace")
                except Exception:
                    preview_str = str(preview)
                print(f"    ❌ 이미지 아님. 응답 미리보기: {preview_str!r}")

        except Exception as e:
            print(f"    ❌ 요청 실패: {e}")

        print()

    print("=" * 50)
    print(f"검증 결과: {tried}건 시도, {success}건 이미지 다운로드 성공")
    if success > 0:
        print("=> 도면 다운로드 파이프라인 구축이 현실적입니다. 다음 단계 진행 가능.")
    else:
        print("=> 직접 다운로드가 막혀 있습니다. 데이터 확보 전략을 재논의해야 합니다.")


if __name__ == "__main__":
    main()
