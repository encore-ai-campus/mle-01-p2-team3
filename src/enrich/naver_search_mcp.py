"""네이버 검색 API를 감싼 MCP 서버 (stdio).

교안 `day05_데이터수집/데이터_수집_1_API_실습파일/1_네이버 API 실습.ipynb` 의 검색 API 호출과
`데이터_수집_3_정적웹크롤링_실습파일` 의 requests + BeautifulSoup 본문 추출을
LLM 에이전트가 도구로 쓸 수 있도록 MCP 도구 4개로 노출한다.

  - naver_local_search : 업체명으로 지역검색 -> 도로명주소 + 업종 카테고리 (업종/주소 확인에 가장 유용)
  - naver_web_search   : 웹문서 검색 -> 제목/요약/링크
  - naver_news_search  : 뉴스 검색 -> 최근 소식으로 업종·소재지 확인
  - fetch_page_text    : 검색으로 찾은 URL의 본문 텍스트를 직접 읽기

단독 실행 확인:  uv run python src/enrich/naver_search_mcp.py --selftest "롯데건설"
MCP 서버로 실행:  uv run python src/enrich/naver_search_mcp.py
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

BASE_URL = "https://openapi.naver.com/v1/search"
TIMEOUT = 10

mcp = FastMCP("naver-search")

_TAG_PAT = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    """네이버 응답의 <b> 강조 태그와 HTML 엔티티를 제거한다."""
    if not text:
        return ""
    return html.unescape(_TAG_PAT.sub("", text)).strip()


def _call(endpoint: str, query: str, display: int, sort: str | None = None) -> dict:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 .env 에 없습니다. "
            "https://developers.naver.com/apps/#/register 에서 검색 API 를 신청하세요."
        )
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    params: dict[str, str | int] = {"query": query, "display": display}
    if sort:
        params["sort"] = sort
    response = requests.get(
        f"{BASE_URL}/{endpoint}.json", headers=headers, params=params, timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def naver_local_search(query: str, display: int = 5) -> str:
    """회사명으로 네이버 지역(업체) 검색. 도로명주소와 업종 카테고리를 돌려준다.

    기업의 소재지와 업종을 확인할 때 가장 먼저 쓸 것.
    Args:
        query: 검색할 회사명 (예: "롯데건설", "주식회사 케이티").
        display: 가져올 결과 수 (최대 5).
    """
    try:
        data = _call("local", query, min(display, 5))
    except Exception as exc:  # noqa: BLE001 - 도구 실패는 에이전트에게 문자열로 알린다
        return f"검색 실패: {exc}"

    items = data.get("items", [])
    if not items:
        return f"'{query}' 지역검색 결과 없음"

    lines = [f"'{query}' 지역검색 결과 {len(items)}건"]
    for i, item in enumerate(items, 1):
        addr = _clean(item.get("roadAddress")) or _clean(item.get("address"))
        lines.append(
            f"{i}. 업체명: {_clean(item.get('title'))}\n"
            f"   업종분류: {_clean(item.get('category'))}\n"
            f"   주소: {addr}\n"
            f"   전화: {_clean(item.get('telephone')) or '-'}"
        )
    return "\n".join(lines)


@mcp.tool()
def naver_web_search(query: str, display: int = 5) -> str:
    """네이버 웹문서 검색. 회사 소개·사업내용을 찾을 때 사용한다.

    Args:
        query: 검색어. "회사명 본사 주소", "회사명 업종" 처럼 구체적으로 쓸 것.
        display: 가져올 결과 수 (최대 10 권장).
    """
    try:
        data = _call("webkr", query, min(display, 10))
    except Exception as exc:  # noqa: BLE001
        return f"검색 실패: {exc}"

    items = data.get("items", [])
    if not items:
        return f"'{query}' 웹문서 검색 결과 없음"

    lines = [f"'{query}' 웹문서 검색 결과 {len(items)}건"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {_clean(item.get('title'))}\n"
            f"   요약: {_clean(item.get('description'))}\n"
            f"   링크: {item.get('link', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def naver_news_search(query: str, display: int = 5) -> str:
    """네이버 뉴스 검색. 웹문서에서 못 찾았을 때 보조로 사용한다.

    Args:
        query: 검색어 (예: "삼성바이오로직스 본사 인천").
        display: 가져올 결과 수 (최대 10 권장).
    """
    try:
        data = _call("news", query, min(display, 10), sort="sim")
    except Exception as exc:  # noqa: BLE001
        return f"검색 실패: {exc}"

    items = data.get("items", [])
    if not items:
        return f"'{query}' 뉴스 검색 결과 없음"

    lines = [f"'{query}' 뉴스 검색 결과 {len(items)}건"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {_clean(item.get('title'))} ({item.get('pubDate', '')})\n"
            f"   요약: {_clean(item.get('description'))}\n"
            f"   링크: {item.get('originallink') or item.get('link', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def fetch_page_text(url: str, max_chars: int = 2000) -> str:
    """검색 결과 링크의 본문 텍스트를 읽는다. 요약만으로 판단이 안 될 때 사용한다.

    Args:
        url: 읽을 페이지 주소 (http/https 만 허용).
        max_chars: 돌려줄 최대 글자 수.
    """
    if not url.startswith(("http://", "https://")):
        return "http/https URL 만 읽을 수 있습니다."
    try:
        from bs4 import BeautifulSoup

        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; corp-enrich/1.0)"},
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\n{2,}", "\n", soup.get_text("\n", strip=True))
    except Exception as exc:  # noqa: BLE001
        return f"페이지 읽기 실패: {exc}"
    return text[:max_chars]


def _selftest(query: str) -> None:
    """키 설정과 네트워크가 정상인지 MCP 없이 바로 확인한다."""
    print("=== naver_local_search ===")
    print(naver_local_search(query))
    print("\n=== naver_web_search ===")
    print(naver_web_search(f"{query} 본사 주소 업종"))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(sys.argv[sys.argv.index("--selftest") + 1])
    else:
        mcp.run()
