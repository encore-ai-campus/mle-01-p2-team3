"""4단계: 기업개요_최종.csv / 종속기업_정리.csv 에 주소·업종을 채운다.

두 파일은 통합 CSV와 달리 아예 컬럼이 비어 있는 항목이 있다.
  - 기업개요_최종.csv : 업종(sicNm) 컬럼이 없음. 주소(enpBsadr)는 이미 다 있음.
  - 종속기업_정리.csv : 주소 컬럼이 없음. 사업내용(sbrdEnpMainBizCtt)은 93건 결측.

먼저 이미 확보한 자료(보강된 통합 CSV, 금융위 기업개요 원본, 2단계 LLM 결과)를
crno·정규화 사명으로 조인해 공짜로 채우고, 그래도 빈 칸만 2단계 작업목록으로 넘긴다.

실행:  uv run python src/enrich/step4_side_csvs.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrich.common import (  # noqa: E402
    CLEAN, FILLED_CORP, FILLED_SUBS, FINAL_CSV, RAW, STEP1_CSV, TODO_OV, TODO_SB,
    is_useful_addr, norm_name, read_jsonl, region_from_addr, write_jsonl,
)

OV_SRC = CLEAN / "기업개요_최종.csv"
SB_SRC = CLEAN / "종속기업_정리.csv"
OV_OUT = CLEAN / "기업개요_최종_보강.csv"
SB_OUT = CLEAN / "종속기업_정리_보강.csv"


def _put(store: dict, key: str, **values: str) -> None:
    if not key:
        return
    slot = store.setdefault(key, {})
    for field, value in values.items():
        if value and not slot.get(field):
            slot[field] = value


def build_lookup() -> tuple[dict, dict]:
    """crno 사전과 정규화 사명 사전을 만든다. 신뢰도 높은 출처부터 넣는다."""
    by_crno: dict[str, dict] = {}
    by_name: dict[str, dict] = {}

    source = FINAL_CSV if FINAL_CSV.exists() else STEP1_CSV
    uni = pd.read_csv(source, dtype=str).fillna("")
    print(f"기준 파일: {source.name} ({len(uni):,}행)")

    for prefix in ("top", "affiliate"):
        sub = uni[uni[f"{prefix}_corpNm"] != ""]
        for _, row in sub.iterrows():
            addr = row[f"{prefix}_addr"]
            values = {
                "addr": addr if is_useful_addr(addr) else "",
                "sicNm": row[f"{prefix}_sicNm"],
            }
            _put(by_crno, row[f"{prefix}_crno"], **values)
            _put(by_name, norm_name(row[f"{prefix}_corpNm"]), **values)

    sub = uni[uni["subsidiary_name"] != ""]
    for _, row in sub.iterrows():
        addr = row["subsidiary_addr"]
        _put(
            by_name,
            norm_name(row["subsidiary_name"]),
            addr=addr if is_useful_addr(addr) else "",
            sicNm=row["subsidiary_bizCtt"],
        )

    # 금융위 기업개요 원본 (주소 보강용)
    files = glob.glob(str(RAW / "기업개요*.csv"))
    if files:
        raw = pd.concat([pd.read_csv(f, dtype=str) for f in files]).fillna("")
        raw = raw.drop_duplicates("crno")
        for _, row in raw.iterrows():
            values = {"addr": row["enpBsadr"], "sicNm": row["sicNm"]}
            _put(by_crno, row["crno"], **values)
            _put(by_name, norm_name(row["corpNm"]), **values)
        print(f"금융위 기업개요 원본 {len(raw):,}건 반영")

    # 2단계 LLM 결과 (근거 URL 있고 low 아닌 것만)
    llm = 0
    for path, name_field in ((FILLED_CORP, "corpNm"), (FILLED_SUBS, "subsidiary_name")):
        for row in read_jsonl(path):
            result = row.get("result", {})
            if result.get("status") != "found" or result.get("confidence") == "low":
                continue
            values = {
                "addr": result.get("addr", ""),
                "sicNm": result.get("sicNm") or result.get("bizCtt", ""),
            }
            if row.get("crno"):
                _put(by_crno, row["crno"], **values)
            _put(by_name, norm_name(row.get(name_field, "")), **values)
            llm += 1
    print(f"LLM 조사 결과 {llm:,}건 반영")
    return by_crno, by_name


def lookup(by_crno: dict, by_name: dict, crno: str, name: str) -> dict:
    """crno 우선, 없으면 사명으로 찾는다."""
    merged = dict(by_name.get(norm_name(name), {}))
    for field, value in (by_crno.get(crno, {}) or {}).items():
        if value:
            merged[field] = value  # crno 일치가 더 정확하므로 우선
    return merged


def enrich_overview(by_crno: dict, by_name: dict) -> int:
    """기업개요_최종.csv 에 sicNm, region 컬럼을 추가한다."""
    df = pd.read_csv(OV_SRC, dtype=str).fillna("")
    df["sicNm"] = ""
    df["region"] = df["enpBsadr"].map(region_from_addr)

    todos = []
    for idx, row in df.iterrows():
        found = lookup(by_crno, by_name, row["crno"], row["corpNm"])
        sic = found.get("sicNm", "")
        if sic:
            df.at[idx, "sicNm"] = sic
        else:
            todos.append({
                "key": row["crno"], "crno": row["crno"], "corpNm": row["corpNm"],
                "need": ["sicNm"], "known_addr": row["enpBsadr"], "known_sicNm": "",
            })

    df.to_csv(OV_OUT, index=False, encoding="utf-8-sig")
    write_jsonl(TODO_OV, todos)
    filled = int((df["sicNm"] != "").sum())
    print(f"\n[기업개요_최종] {len(df):,}행")
    print(f"  업종 채움 {filled:,} / 남은 결측 {len(todos):,}")
    print(f"  지역 채움 {int((df['region'] != '').sum()):,}")
    print(f"  저장: {OV_OUT.name}")
    return len(todos)


def enrich_subsidiary(by_crno: dict, by_name: dict) -> int:
    """종속기업_정리.csv 에 addr, region, sicNm 컬럼을 추가한다."""
    df = pd.read_csv(SB_SRC, dtype=str).fillna("")
    df["addr"] = ""
    df["region"] = ""
    df["sicNm"] = ""

    todos: dict[str, dict] = {}
    for idx, row in df.iterrows():
        name = row["sbrdEnpNm"]
        # crno 는 모회사 번호라 종속기업 조회에 쓰면 안 된다. 사명으로만 찾는다.
        found = by_name.get(norm_name(name), {})
        addr = found.get("addr", "")
        if addr:
            df.at[idx, "addr"] = addr
            df.at[idx, "region"] = region_from_addr(addr)
        sic = found.get("sicNm", "") or row["sbrdEnpMainBizCtt"]
        if sic:
            df.at[idx, "sicNm"] = sic
        if not addr or not sic:
            need = ([] if addr else ["region", "addr"]) + ([] if sic else ["bizCtt"])
            key = norm_name(name) or name
            todos.setdefault(key, {
                "key": key, "subsidiary_name": name, "need": need,
                "known_addr": addr, "known_bizCtt": row["sbrdEnpMainBizCtt"],
                "parent_corpNm": "",
            })

    df.to_csv(SB_OUT, index=False, encoding="utf-8-sig")
    write_jsonl(TODO_SB, list(todos.values()))
    print(f"\n[종속기업_정리] {len(df):,}행 (유니크 사명 {df['sbrdEnpNm'].nunique():,})")
    print(f"  주소 채움 {int((df['addr'] != '').sum()):,}행 / 지역 {int((df['region'] != '').sum()):,}행")
    print(f"  업종 채움 {int((df['sicNm'] != '').sum()):,}행")
    print(f"  남은 유니크 결측 {len(todos):,}건")
    print(f"  저장: {SB_OUT.name}")
    return len(todos)


def main() -> None:
    by_crno, by_name = build_lookup()
    print(f"조회 사전: crno {len(by_crno):,}건 / 사명 {len(by_name):,}건")
    n_ov = enrich_overview(by_crno, by_name)
    n_sb = enrich_subsidiary(by_crno, by_name)
    print(f"\n2단계로 넘길 작업: {TODO_OV.name} {n_ov:,}건 / {TODO_SB.name} {n_sb:,}건")


if __name__ == "__main__":
    main()
