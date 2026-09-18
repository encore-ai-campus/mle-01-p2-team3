from collections import defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path.home() / "projects" / "mle-01-p2-team3"
DATA_DIR = PROJECT_ROOT / "data" / "clean"

# URL이 있는 원본 뉴스 파일
input_path = DATA_DIR / "news_articles.jsonl"

# 기업명 표기를 가져올 기업개요 파일
company_path = DATA_DIR / "기업개요_최종.csv"

# 임베딩 담당자에게 전달할 최종 파일
output_path = DATA_DIR / "company_news_for_embedding.jsonl"


def format_date(date_text: str) -> str:
    """Thu, 17 Sep 2026 ... 형식을 2026-09-17로 바꾼다."""
    try:
        return parsedate_to_datetime(date_text).date().isoformat()
    except Exception:
        return date_text


# 1. 기업개요에서 crno -> 기업명 사전 만들기
import re


def clean_company_name(name: str) -> str:
    name = str(name).strip()

    for token in [
        "(주)",
        "(유)",
        "㈜",
        "㈔",
        "주식회사",
        "유한회사",
    ]:
        name = name.replace(token, "")

    return re.sub(r"\s+", " ", name).strip()


company_df = pd.read_csv(
    company_path,
    dtype={"crno": str},
    encoding="utf-8-sig",
)

company_name_map = {
    row["crno"]: clean_company_name(row["corpNm"])
    for _, row in company_df.iterrows()
}


# 2. 기사 380건을 company_id별로 묶기
company_news_map = defaultdict(list)
seen_urls = defaultdict(set)

with input_path.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        row = json.loads(line)

        crno = row["company_crno"]
        url = row["original_url"]

        # 같은 기업에 같은 링크가 중복되면 한 번만 보관
        if url in seen_urls[crno]:
            continue

        seen_urls[crno].add(url)

        company_news_map[crno].append(
            {
                "title": row["title"],
                "summary": row["description"],
                "url": url,
                "published_at": format_date(row["published_at"]),
            }
        )


# 3. 기업별 뉴스 목록과 embedding_text 만들기
company_rows = []

for crno, news_list in company_news_map.items():
    company_name = company_name_map.get(crno, "")

    # 기업당 최대 5개만 사용
    news_list = news_list[:5]

    text_parts = [f"기업명: {company_name}"]

    for index, news in enumerate(news_list, start=1):
        text_parts.append(f"뉴스{index} 제목: {news['title']}")
        text_parts.append(f"뉴스{index} 요약: {news['summary']}")

    embedding_text = "\n".join(text_parts)

    company_rows.append(
        {
            "company_id": f"parent_company:{crno}",
            "company_name": company_name,
            "news": news_list,
            "embedding_text": embedding_text,
        }
    )


# 4. JSONL 저장: 한 줄이 기업 한 개
with output_path.open("w", encoding="utf-8") as file:
    for row in company_rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


# 5. 결과 확인
news_counts = [len(row["news"]) for row in company_rows]

print(f"저장 완료: {output_path}")
print(f"기업 수: {len(company_rows)}개")
print(f"뉴스 수 합계: {sum(news_counts)}건")
print(f"기업당 뉴스 최소/최대: {min(news_counts)} / {max(news_counts)}건")

print("\n샘플:")
print(json.dumps(company_rows[0], ensure_ascii=False, indent=2))