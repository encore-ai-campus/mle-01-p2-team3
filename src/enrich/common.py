"""보강 파이프라인 공용 유틸 (경로, 지역 파싱, 사명 정규화, JSONL 입출력)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"

SRC_CSV = CLEAN / "모기업_계열사_종속기업_통합.csv"
STEP1_CSV = CLEAN / "모기업_계열사_종속기업_통합_1차보강.csv"
FINAL_CSV = CLEAN / "모기업_계열사_종속기업_통합_보강.csv"

TODO_CORP = WORK / "todo_corp.jsonl"
TODO_SUBS = WORK / "todo_subs.jsonl"
FILLED_CORP = WORK / "filled_corp.jsonl"
FILLED_SUBS = WORK / "filled_subs.jsonl"

# 4단계에서 다루는 개별 CSV (기업개요_최종 / 종속기업_정리)
TODO_OV = WORK / "todo_ov.jsonl"
TODO_SB = WORK / "todo_sb.jsonl"
FILLED_OV = WORK / "filled_ov.jsonl"
FILLED_SB = WORK / "filled_sb.jsonl"
CHANGELOG = WORK / "보강_변경내역.csv"

# 원본 CSV의 region 표기 규칙(축약형)과 동일하게 맞춘다.
REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

# 긴 표기를 먼저 매칭해야 하므로 순서 유지.
_REGION_PREFIXES: list[tuple[str, str]] = [
    ("서울특별시", "서울"), ("서울시", "서울"), ("서울", "서울"),
    ("부산광역시", "부산"), ("부산시", "부산"), ("부산", "부산"),
    ("대구광역시", "대구"), ("대구시", "대구"), ("대구", "대구"),
    ("인천광역시", "인천"), ("인천시", "인천"), ("인천", "인천"),
    ("광주광역시", "광주"), ("광주시", "광주"), ("광주", "광주"),
    ("대전광역시", "대전"), ("대전시", "대전"), ("대전", "대전"),
    ("울산광역시", "울산"), ("울산시", "울산"), ("울산", "울산"),
    ("세종특별자치시", "세종"), ("세종시", "세종"), ("세종", "세종"),
    ("경기도", "경기"), ("경기", "경기"),
    ("강원특별자치도", "강원"), ("강원도", "강원"), ("강원", "강원"),
    ("충청북도", "충북"), ("충북", "충북"),
    ("충청남도", "충남"), ("충남", "충남"),
    ("전북특별자치도", "전북"), ("전라북도", "전북"), ("전북", "전북"),
    ("전라남도", "전남"), ("전남", "전남"),
    ("경상북도", "경북"), ("경북", "경북"),
    ("경상남도", "경남"), ("경남", "경남"),
    ("제주특별자치도", "제주"), ("제주도", "제주"), ("제주", "제주"),
]

# 주소 자리에 들어있지만 지역 정보가 없는 값들.
_USELESS_ADDR = {"", "-", "대한민국", "한국", "국내", "korea", "south korea"}


def region_from_addr(addr: str | None) -> str:
    """주소 문자열 앞부분에서 시·도 축약명을 뽑는다. 못 뽑으면 빈 문자열."""
    if not addr:
        return ""
    text = str(addr).strip()
    if text.lower() in _USELESS_ADDR:
        return ""
    # '대한민국 서울특별시 ...' 처럼 국가명이 앞에 붙은 경우 제거
    for lead in ("대한민국", "한국"):
        if text.startswith(lead):
            text = text[len(lead):].strip()
    for prefix, region in _REGION_PREFIXES:
        if text.startswith(prefix):
            return region
    return ""


def is_useful_addr(addr: str | None) -> bool:
    """'대한민국'처럼 의미 없는 주소인지 판별."""
    if not addr:
        return False
    return str(addr).strip().lower() not in _USELESS_ADDR


_SUFFIX_PAT = re.compile(
    r"(주식회사|유한회사|유한책임회사|합자회사|합명회사|\(주\)|\(유\)|㈜|㈜|\(株\))"
)
_NON_WORD = re.compile(r"[^0-9A-Za-z가-힣]")


def norm_name(name: str | None) -> str:
    """사명 정규화: 법인격 표기·공백·특수문자 제거 후 소문자화."""
    if not name:
        return ""
    text = _SUFFIX_PAT.sub("", str(name))
    text = _NON_WORD.sub("", text)
    return text.lower()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 실행 중 중단되면 마지막 줄이 잘려 있을 수 있다. 그 줄만 버린다.
                print(f"  [경고] {path.name} {line_no}번째 줄을 읽지 못해 건너뜀")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
