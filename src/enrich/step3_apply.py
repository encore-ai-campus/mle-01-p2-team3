"""3단계: 2단계 LLM 결과를 검증해 1차보강 CSV에 반영하고 최종본을 만든다.

  - status 가 found 이고 근거 URL이 있는 값만 반영
  - confidence 가 기준 미만이면 반영하지 않음 (기본: low 제외)
  - region 은 허용 목록, 주소는 실제 주소 형태인지 다시 확인
  - 무엇을 무엇으로 바꿨는지 근거 URL과 함께 변경내역 CSV로 남김

실행:  uv run python src/enrich/step3_apply.py --min-confidence medium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrich.common import (  # noqa: E402
    FILLED_CORP, FILLED_SUBS, FINAL_CSV, REGIONS, STEP1_CSV, WORK,
    is_useful_addr, norm_name, read_jsonl, region_from_addr,
)

APPLY_LOG = WORK / "보강_LLM_적용내역.csv"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

applied: list[dict[str, str]] = []


def usable(result: dict, min_rank: int) -> bool:
    """반영해도 되는 결과인지 검사한다."""
    if result.get("status") != "found":
        return False
    if not str(result.get("evidence_url", "")).startswith("http"):
        return False
    return CONFIDENCE_RANK.get(result.get("confidence", "low"), 0) >= min_rank


def clean_value(field: str, value: str) -> str:
    """필드별 값 형식을 마지막으로 한 번 더 검사한다."""
    value = str(value or "").strip()
    if not value:
        return ""
    if field == "region":
        return value if value in REGIONS else ""
    if field in ("addr",):
        # 주소라면 시·도로 시작하거나 최소한 '시/군/구'가 들어가야 한다.
        if not is_useful_addr(value):
            return ""
        if not region_from_addr(value) and not any(x in value for x in ("시 ", "군 ", "구 ")):
            return ""
    if field in ("sicNm", "bizCtt") and len(value) > 60:
        return ""
    if field == "domestic":
        return value if value in ("국내", "해외") else ""
    return value


def log(idx: int, key: str, column: str, before: str, after: str, result: dict) -> None:
    applied.append({
        "row": str(idx), "key": key, "column": column,
        "before": before, "after": after,
        "confidence": result.get("confidence", ""),
        "evidence_url": result.get("evidence_url", ""),
        "evidence": str(result.get("evidence", ""))[:120],
    })


def apply_corp(df: pd.DataFrame, min_rank: int) -> None:
    by_key: dict[str, dict] = {}
    for row in read_jsonl(FILLED_CORP):
        result = row.get("result", {})
        if usable(result, min_rank):
            by_key[row["key"]] = result
    print(f"[법인] 반영 가능한 결과 {len(by_key):,}건")
    if not by_key:
        return

    for prefix in ("top", "affiliate"):
        crno_col, name_col = f"{prefix}_crno", f"{prefix}_corpNm"
        sub = df[df[name_col] != ""]
        for idx, row in sub.iterrows():
            key = row[crno_col] or f"name:{norm_name(row[name_col])}"
            result = by_key.get(key)
            if not result:
                continue
            for field, column in (("addr", f"{prefix}_addr"),
                                  ("region", f"{prefix}_region"),
                                  ("sicNm", f"{prefix}_sicNm")):
                if df.at[idx, column] != "":
                    continue
                value = clean_value(field, result.get(field, ""))
                if not value:
                    continue
                df.at[idx, column] = value
                log(idx, key, column, "", value, result)
            # 주소만 찾고 지역을 비워 둔 경우 주소에서 파생
            region_col = f"{prefix}_region"
            if df.at[idx, region_col] == "":
                region = region_from_addr(df.at[idx, f"{prefix}_addr"])
                if region:
                    df.at[idx, region_col] = region
                    log(idx, key, region_col, "", region, result)


def apply_subs(df: pd.DataFrame, min_rank: int) -> None:
    by_key: dict[str, dict] = {}
    for row in read_jsonl(FILLED_SUBS):
        result = row.get("result", {})
        if usable(result, min_rank):
            by_key[row["key"]] = result
    print(f"[종속기업] 반영 가능한 결과 {len(by_key):,}건")
    if not by_key:
        return

    sub = df[df["subsidiary_name"] != ""]
    for idx, row in sub.iterrows():
        result = by_key.get(norm_name(row["subsidiary_name"]))
        if not result:
            continue
        addr = clean_value("addr", result.get("addr", ""))
        if addr and not is_useful_addr(df.at[idx, "subsidiary_addr"]):
            before = df.at[idx, "subsidiary_addr"]
            df.at[idx, "subsidiary_addr"] = addr
            log(idx, row["subsidiary_name"], "subsidiary_addr", before, addr, result)
        for field, column in (("region", "subsidiary_region"), ("bizCtt", "subsidiary_bizCtt")):
            if df.at[idx, column] != "":
                continue
            value = clean_value(field, result.get(field, ""))
            if value:
                df.at[idx, column] = value
                log(idx, row["subsidiary_name"], column, "", value, result)
        if df.at[idx, "subsidiary_region"] == "":
            region = region_from_addr(df.at[idx, "subsidiary_addr"])
            if region:
                df.at[idx, "subsidiary_region"] = region
                log(idx, row["subsidiary_name"], "subsidiary_region", "", region, result)
        domestic = clean_value("domestic", result.get("domestic", ""))
        if domestic and df.at[idx, "domestic"] in ("", "모름"):
            before = df.at[idx, "domestic"]
            df.at[idx, "domestic"] = domestic
            log(idx, row["subsidiary_name"], "domestic", before, domestic, result)


def final_cross_fill(df: pd.DataFrame) -> int:
    """LLM이 새로 채운 값을 같은 회사의 다른 등장 위치로 한 번 더 전파한다.

    예: 어떤 회사의 업종을 법인 칸에서 알아냈는데, 같은 회사가 다른 행에서
    종속기업으로도 등장한다면 그쪽 주소·사업내용도 같이 메워진다.
    """
    from enrich import step1_rule_fill as step1

    step1.changes.clear()
    step1.rule3_cross_fill(df)
    step1.rule1_region_from_addr(df)
    for change in step1.changes:
        applied.append({
            "row": change["row"], "key": change["key"], "column": change["column"],
            "before": change["before"], "after": change["after"],
            "confidence": "", "evidence_url": "", "evidence": f"({change['source']})",
        })
    return len(step1.changes)


def gap_report(df: pd.DataFrame, title: str) -> dict[str, int]:
    cols = ["top_region", "top_addr", "top_sicNm",
            "affiliate_region", "affiliate_addr", "affiliate_sicNm",
            "subsidiary_region", "subsidiary_addr", "subsidiary_bizCtt"]
    stat = {}
    for col in cols:
        owner = "subsidiary_name" if col.startswith("subsidiary") else col.split("_")[0] + "_corpNm"
        scope = df[df[owner] != ""]
        stat[col] = int((scope[col] == "").sum())
    subs = df[df["subsidiary_name"] != ""]
    stat["subsidiary_addr_무의미"] = int(
        (subs["subsidiary_addr"] != "").sum()
        - subs["subsidiary_addr"].map(is_useful_addr).sum()
    )
    stat["domestic_모름"] = int((df["domestic"] == "모름").sum())
    print(f"\n[{title}]")
    for col, n in stat.items():
        print(f"  {col:<20} {n:>6,}")
    return stat


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 보강 결과를 검증 후 반영")
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--no-cross-fill", action="store_true",
                        help="LLM 결과를 같은 회사의 다른 행으로 재전파하지 않는다")
    args = parser.parse_args()
    min_rank = CONFIDENCE_RANK[args.min_confidence]

    if not STEP1_CSV.exists():
        raise SystemExit(f"{STEP1_CSV.name} 이 없습니다. step1_rule_fill.py 를 먼저 실행하세요.")

    df = pd.read_csv(STEP1_CSV, dtype=str).fillna("")
    before = gap_report(df, "1차보강 상태")

    print(f"\nLLM 결과 반영 (confidence >= {args.min_confidence})")
    apply_corp(df, min_rank)
    apply_subs(df, min_rank)

    if not args.no_cross_fill:
        n = final_cross_fill(df)
        print(f"[재전파] LLM 결과를 같은 회사의 다른 행으로 {n:,}칸 추가 전파")

    after = gap_report(df, "최종 상태")
    print("\n필드별 감소량")
    for col in before:
        delta = before[col] - after[col]
        if delta:
            print(f"  {col:<20} -{delta:,}  ({before[col]:,} -> {after[col]:,})")

    df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(applied).to_csv(APPLY_LOG, index=False, encoding="utf-8-sig")
    print(f"\n저장: {FINAL_CSV.name} (LLM 반영 {len(applied):,}건)")
    print(f"적용내역(근거 URL 포함): {APPLY_LOG.name}")


if __name__ == "__main__":
    main()
