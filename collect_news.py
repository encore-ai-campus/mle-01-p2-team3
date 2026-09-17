import hashlib
import html
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

PROJECT_ROOT = Path.home() / "projects" / "mle-01-p2-team3"
DATA_DIR = PROJECT_ROOT / "data" / "clean"

TARGET_FILE = DATA_DIR / "news_target_companies_top100.csv"
OUTPUT_FILE = DATA_DIR / "news_articles.jsonl"

NEWS_PER_COMPANY = 5
DISPLAY_COUNT = 100

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# 텍스트·회사명 정리 함수
# --------------------------------------------------

def clean_text(text: str) -> str:
    """뉴스 API 응답에 포함된 HTML 태그와 엔티티를 제거한다."""
    text = re.sub(r"<[^>]+>", "", str(text))
    return html.unescape(text).strip()


def make_search_name(company_name: str) -> str:
    """(주), 주식회사 등을 제거해 뉴스 검색에 적합한 이름으로 만든다."""
    company_name = str(company_name)

    for token in ["(주)", "㈜", "주식회사", "유한회사"]:
        company_name = company_name.replace(token, "")

    return company_name.strip()


def is_direct_company_news(item: dict, search_name: str) -> bool:
    """
    제목에 정확한 기업명이 들어간 기사만 남긴다.
    단순 언급 기사·부동산 기사 등을 줄이기 위한 1차 필터다.
    """
    title = clean_text(item.get("title", ""))
    return search_name in title


def make_record_id(company_crno: str, original_url: str) -> str:
    """같은 기업-기사 조합을 구분하기 위한 고유 ID."""
    raw = f"{company_crno}|{original_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_news_item(
    item: dict,
    company_name: str,
    company_crno: str,
    search_name: str,
) -> dict:
    """API 응답 하나를 JSONL 저장용 뉴스 문서로 변환한다."""
    original_url = item.get("originallink") or item.get("link", "")
    publisher = urlparse(original_url).netloc.removeprefix("www.")

    return {
        "record_id": make_record_id(company_crno, original_url),
        "company_name": company_name,
        "company_crno": company_crno,
        "search_query": search_name,
        "title": clean_text(item.get("title", "")),
        "description": clean_text(item.get("description", "")),
        "published_at": item.get("pubDate", ""),
        "publisher": publisher,
        "original_url": original_url,
        "naver_url": item.get("link", ""),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------
# 네이버 뉴스 API 호출
# --------------------------------------------------

def fetch_news(search_name: str) -> list[dict]:
    params = {
        "query": search_name,
        "display": DISPLAY_COUNT,
        "start": 1,
        "sort": "date",
    }

    url = "https://openapi.naver.com/v1/search/news.json?" + urlencode(params)

    request = Request(
        url,
        headers={
            "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
            "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
        },
    )

    with urlopen(request) as response:
        result = json.load(response)

    return result.get("items", [])


# --------------------------------------------------
# 기존 JSONL 읽기
# --------------------------------------------------

existing_records = []
existing_ids = set()
existing_counts = Counter()

if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            existing_records.append(record)
            existing_ids.add(record["record_id"])
            existing_counts[str(record["company_crno"])] += 1

print(f"기존 저장 뉴스: {len(existing_records)}건")


# --------------------------------------------------
# 상위 100개 기업 뉴스 수집
# --------------------------------------------------

company_df = pd.read_csv(
    TARGET_FILE,
    dtype={"crno": str},
)

saved_count = 0
failed_companies = []

with OUTPUT_FILE.open("a", encoding="utf-8") as output_file:
    for index, row in company_df.iterrows():
        company_name = str(row["company_name"])
        company_crno = str(row["crno"])
        search_name = make_search_name(company_name)

        already_saved = existing_counts[company_crno]

        if already_saved >= NEWS_PER_COMPANY:
            print(
                f"[{index + 1}/{len(company_df)}] "
                f"{company_name}: 이미 {already_saved}건 저장됨 → 건너뜀"
            )
            continue

        print(
            f"[{index + 1}/{len(company_df)}] "
            f"{company_name} 검색 중..."
        )

        try:
            items = fetch_news(search_name)

        except HTTPError as error:
            error_message = error.read().decode("utf-8")
            print(f"  API 오류 {error.code}: {error_message}")
            failed_companies.append(company_name)
            continue

        except Exception as error:
            print(f"  수집 오류: {error}")
            failed_companies.append(company_name)
            continue

        company_news = []

        for item in items:
            if not is_direct_company_news(item, search_name):
                continue

            record = normalize_news_item(
                item=item,
                company_name=company_name,
                company_crno=company_crno,
                search_name=search_name,
            )

            if record["record_id"] in existing_ids:
                continue

            if not record["title"] or not record["description"]:
                continue

            company_news.append(record)

            if already_saved + len(company_news) >= NEWS_PER_COMPANY:
                break

        for record in company_news:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            existing_ids.add(record["record_id"])
            existing_counts[company_crno] += 1
            saved_count += 1

        print(
            f"  새로 저장: {len(company_news)}건 / "
            f"총 저장: {existing_counts[company_crno]}건"
        )

        # API에 너무 빠르게 연속 요청하지 않도록 잠깐 대기
        time.sleep(0.1)


# --------------------------------------------------
# 수집 결과 요약
# --------------------------------------------------

print("\n수집 완료")
print(f"이번 실행에서 추가 저장: {saved_count}건")
print(f"최종 파일: {OUTPUT_FILE}")

not_enough = [
    company_name
    for _, row in company_df.iterrows()
    if existing_counts[str(row['crno'])] < NEWS_PER_COMPANY
    for company_name in [row["company_name"]]
]

print(f"뉴스 5건 미만 기업 수: {len(not_enough)}개")

if not_enough:
    print("뉴스가 부족한 기업 예시:")
    for company_name in not_enough[:20]:
        print("-", company_name)

if failed_companies:
    print("\nAPI 호출 실패 기업:")
    for company_name in failed_companies:
        print("-", company_name)