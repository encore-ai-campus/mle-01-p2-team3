import html
import json
import os
import re
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv


# 검색할 기업 또는 그룹명
TARGET_COMPANY = "한화에어로스페이스"


def clean_text(text: str) -> str:
    """네이버 응답의 HTML 강조 태그와 엔티티를 제거한다."""
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def is_direct_target_news(item: dict) -> bool:
    """
    제목에 대상 기업명이 직접 들어간 기사만 남긴다.
    예: '한화'라면 한화생명·한화에어로스페이스 등 그룹 기사도 포함된다.
    """
    title = clean_text(item.get("title", ""))
    return TARGET_COMPANY in title


def normalize_item(item: dict) -> dict:
    """벡터 검색에 사용할 뉴스 문서 형태로 정리한다."""
    original_url = item.get("originallink") or item.get("link", "")
    publisher = urlparse(original_url).netloc.removeprefix("www.")

    return {
        "company_name": TARGET_COMPANY,
        "title": clean_text(item.get("title", "")),
        "description": clean_text(item.get("description", "")),
        "published_at": item.get("pubDate", ""),
        "publisher": publisher,
        "original_url": original_url,
        "naver_url": item.get("link", ""),
    }


load_dotenv()

params = {
    "query": TARGET_COMPANY,
    "display": 100,
    "start": 1,
    "sort": "date",
}

# 기존 Naver Developers 뉴스 검색 API
url = "https://openapi.naver.com/v1/search/news.json?" + urlencode(params)

request = Request(
    url,
    headers={
        "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
    },
)

try:
    with urlopen(request) as response:
        result = json.load(response)

except HTTPError as error:
    print("HTTP status:", error.code)
    print("API message:", error.read().decode("utf-8"))
    raise


filtered_news = []
seen_urls = set()

for item in result["items"]:
    if not is_direct_target_news(item):
        continue

    article = normalize_item(item)

    # 같은 원문 기사는 하나만 유지
    if article["original_url"] in seen_urls:
        continue

    seen_urls.add(article["original_url"])
    filtered_news.append(article)


# 최신 관련 기사 10개만 사용
top_news = filtered_news[:10]

print(f"\n{TARGET_COMPANY} 관련 기사 수: {len(top_news)}건\n")

for index, article in enumerate(top_news, start=1):
    print(f"[{index}] {article['published_at']}")
    print(f"제목: {article['title']}")
    print(f"요약: {article['description']}")
    print(f"언론사: {article['publisher']}")
    print(f"링크: {article['original_url']}")
    print()


# LLM 답변에 넣을 수 있는 Markdown 근거 링크 형식
print("LLM 근거 기사 형식:\n")

for article in top_news:
    print(
        f"- [{article['title']}]({article['original_url']}) "
        f"— {article['publisher']}, {article['published_at']}"
    )