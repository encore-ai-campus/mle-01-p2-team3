"""1단계: LLM 없이 규칙만으로 채울 수 있는 결측을 먼저 메운다.

  R1. 주소 문자열에서 시·도(region) 파싱
  R2. data/raw 의 금융위 기업개요(getCorpOutline_V2) 원본을 crno로 재조인
  R3. 사명 정규화 매칭으로 법인 <-> 종속기업 정보 교차 전파

남은 결측은 2단계(웹검색 MCP + LLM) 입력으로 data/work/todo_*.jsonl 에 떨군다.

실행:  uv run python src/enrich/step1_rule_fill.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrich.common import (  # noqa: E402
    CHANGELOG, RAW, SRC_CSV, STEP1_CSV, TODO_CORP, TODO_SUBS,
    is_useful_addr, norm_name, region_from_addr, write_jsonl,
)

CORP_PREFIXES = ("top", "affiliate")
changes: list[dict[str, str]] = []


def record(idx: int, key: str, column: str, before: str, after: str, rule: str) -> None:
    changes.append({
        "row": str(idx), "key": key, "column": column,
        "before": before, "after": after, "source": rule,
    })


def rule1_region_from_addr(df: pd.DataFrame) -> None:
    """주소가 있는데 region이 비어 있으면 주소에서 파싱한다."""
    pairs = [(f"{p}_addr", f"{p}_region") for p in CORP_PREFIXES]
    pairs.append(("subsidiary_addr", "subsidiary_region"))
    for addr_col, region_col in pairs:
        target = df[(df[region_col] == "") & (df[addr_col] != "")]
        for idx, row in target.iterrows():
            region = region_from_addr(row[addr_col])
            if region:
                df.at[idx, region_col] = region
                record(idx, row[addr_col][:30], region_col, "", region, "R1_주소파싱")


def rule2_rejoin_raw_outline(df: pd.DataFrame) -> None:
    """금융위 기업개요 원본(data/raw/기업개요*.csv)을 crno로 다시 붙인다."""
    files = glob.glob(str(RAW / "기업개요*.csv"))
    if not files:
        print("  [R2] data/raw 에 기업개요 원본이 없어 건너뜀")
        return
    raw = (
        pd.concat([pd.read_csv(f, dtype=str) for f in files])
        .fillna("")
        .drop_duplicates("crno")
        .set_index("crno")
    )
    print(f"  [R2] 기업개요 원본 {len(raw):,}개 법인 로드")
    for prefix in CORP_PREFIXES:
        crno_col, addr_col = f"{prefix}_crno", f"{prefix}_addr"
        region_col, sic_col = f"{prefix}_region", f"{prefix}_sicNm"
        target = df[(df[addr_col] == "") & (df[crno_col] != "")]
        for idx, row in target.iterrows():
            crno = row[crno_col]
            if crno not in raw.index:
                continue
            hit = raw.loc[crno]
            addr = str(hit.get("enpBsadr", "")).strip()
            if is_useful_addr(addr):
                df.at[idx, addr_col] = addr
                record(idx, crno, addr_col, "", addr, "R2_기업개요API")
                region = region_from_addr(addr)
                if region and df.at[idx, region_col] == "":
                    df.at[idx, region_col] = region
                    record(idx, crno, region_col, "", region, "R2_기업개요API")
            sic = str(hit.get("sicNm", "")).strip()
            if sic and df.at[idx, sic_col] == "":
                df.at[idx, sic_col] = sic
                record(idx, crno, sic_col, "", sic, "R2_기업개요API")


def _collect_known(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """정규화 사명 -> {addr, region, sicNm, bizCtt, domestic} 사전을 만든다."""
    known: dict[str, dict[str, str]] = {}

    def put(name: str, **values: str) -> None:
        key = norm_name(name)
        if not key:
            return
        slot = known.setdefault(key, {})
        for field, value in values.items():
            if value and not slot.get(field):
                slot[field] = value

    for prefix in CORP_PREFIXES:
        sub = df[df[f"{prefix}_corpNm"] != ""]
        for _, row in sub.iterrows():
            addr = row[f"{prefix}_addr"]
            put(
                row[f"{prefix}_corpNm"],
                addr=addr if is_useful_addr(addr) else "",
                region=row[f"{prefix}_region"],
                sicNm=row[f"{prefix}_sicNm"],
            )
    sub = df[df["subsidiary_name"] != ""]
    for _, row in sub.iterrows():
        addr = row["subsidiary_addr"]
        put(
            row["subsidiary_name"],
            addr=addr if is_useful_addr(addr) else "",
            region=row["subsidiary_region"],
            bizCtt=row["subsidiary_bizCtt"],
            domestic=row["domestic"] if row["domestic"] != "모름" else "",
        )
    return known


def rule3_cross_fill(df: pd.DataFrame) -> None:
    """같은 회사가 법인 행과 종속기업 행에 모두 등장하면 정보를 서로 채워준다."""
    known = _collect_known(df)
    print(f"  [R3] 정규화 사명 사전 {len(known):,}건")

    for prefix in CORP_PREFIXES:
        name_col = f"{prefix}_corpNm"
        sub = df[df[name_col] != ""]
        for idx, row in sub.iterrows():
            slot = known.get(norm_name(row[name_col]))
            if not slot:
                continue
            for field, column in (("addr", f"{prefix}_addr"),
                                  ("region", f"{prefix}_region"),
                                  ("sicNm", f"{prefix}_sicNm")):
                value = slot.get(field, "")
                if value and df.at[idx, column] == "":
                    df.at[idx, column] = value
                    record(idx, row[name_col], column, "", value, "R3_사명교차전파")

    sub = df[df["subsidiary_name"] != ""]
    for idx, row in sub.iterrows():
        slot = known.get(norm_name(row["subsidiary_name"]))
        if not slot:
            continue
        addr = slot.get("addr", "")
        if addr and not is_useful_addr(df.at[idx, "subsidiary_addr"]):
            before = df.at[idx, "subsidiary_addr"]
            df.at[idx, "subsidiary_addr"] = addr
            record(idx, row["subsidiary_name"], "subsidiary_addr", before, addr, "R3_사명교차전파")
        for field, column in (("region", "subsidiary_region"), ("bizCtt", "subsidiary_bizCtt")):
            value = slot.get(field, "")
            if value and df.at[idx, column] == "":
                df.at[idx, column] = value
                record(idx, row["subsidiary_name"], column, "", value, "R3_사명교차전파")
        # 국내 주소가 확인되면 domestic 모름 -> 국내 로 확정
        if df.at[idx, "domestic"] == "모름" and df.at[idx, "subsidiary_region"] != "":
            df.at[idx, "domestic"] = "국내"
            record(idx, row["subsidiary_name"], "domestic", "모름", "국내", "R3_지역확인")


def build_todos(df: pd.DataFrame) -> tuple[int, int]:
    """남은 결측을 2단계 LLM 작업 목록으로 저장한다."""
    corp_rows: dict[str, dict] = {}
    for prefix in CORP_PREFIXES:
        sub = df[df[f"{prefix}_corpNm"] != ""]
        for _, row in sub.iterrows():
            crno = row[f"{prefix}_crno"]
            name = row[f"{prefix}_corpNm"]
            need = [f for f, col in (("addr", f"{prefix}_addr"), ("sicNm", f"{prefix}_sicNm"))
                    if row[col] == ""]
            if not need:
                continue
            key = crno or f"name:{norm_name(name)}"
            slot = corp_rows.setdefault(key, {
                "key": key, "crno": crno, "corpNm": name, "need": [],
                "known_addr": row[f"{prefix}_addr"], "known_sicNm": row[f"{prefix}_sicNm"],
            })
            slot["need"] = sorted(set(slot["need"]) | set(need))

    subs_rows: dict[str, dict] = {}
    sub = df[df["subsidiary_name"] != ""]
    for _, row in sub.iterrows():
        need = []
        if row["subsidiary_region"] == "":
            need.append("region")
        if row["subsidiary_bizCtt"] == "":
            need.append("bizCtt")
        if row["domestic"] in ("", "모름"):
            need.append("domestic")
        if not need:
            continue
        key = norm_name(row["subsidiary_name"]) or row["subsidiary_name"]
        slot = subs_rows.setdefault(key, {
            "key": key, "subsidiary_name": row["subsidiary_name"], "need": [],
            "known_addr": row["subsidiary_addr"], "known_bizCtt": row["subsidiary_bizCtt"],
            "parent_corpNm": row["top_corpNm"],
        })
        slot["need"] = sorted(set(slot["need"]) | set(need))

    write_jsonl(TODO_CORP, list(corp_rows.values()))
    write_jsonl(TODO_SUBS, list(subs_rows.values()))
    return len(corp_rows), len(subs_rows)


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
    print(f"\n[{title}] 해당 없는 행(구조적 공백)은 제외한 실결측")
    for col, n in stat.items():
        print(f"  {col:<20} {n:>6,}")
    return stat


def main() -> None:
    df = pd.read_csv(SRC_CSV, dtype=str).fillna("")
    df = df.apply(lambda s: s.str.strip())
    print(f"원본 {len(df):,}행 로드: {SRC_CSV.name}")
    before = gap_report(df, "보강 전")

    print("\n규칙 적용")
    rule1_region_from_addr(df)
    rule2_rejoin_raw_outline(df)
    rule3_cross_fill(df)
    rule1_region_from_addr(df)  # R2/R3로 주소가 새로 생긴 행 재파싱

    after = gap_report(df, "1단계 보강 후")
    print("\n필드별 감소량")
    for col in before:
        delta = before[col] - after[col]
        if delta:
            print(f"  {col:<20} -{delta:,}  ({before[col]:,} -> {after[col]:,})")

    df.to_csv(STEP1_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(CHANGELOG, index=False, encoding="utf-8-sig")
    n_corp, n_subs = build_todos(df)
    print(f"\n저장: {STEP1_CSV.name}  (변경 {len(changes):,}건)")
    print(f"변경내역: {CHANGELOG.name}")
    print(f"2단계 작업목록: 법인 {n_corp:,}건 -> {TODO_CORP.name} / 종속기업 {n_subs:,}건 -> {TODO_SUBS.name}")


if __name__ == "__main__":
    main()
