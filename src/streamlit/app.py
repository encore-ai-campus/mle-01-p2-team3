from __future__ import annotations

import base64
import html
import importlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

from aura_graph import build_graph_elements, fetch_graph_paths, search_companies

SECTIONS = ["Overview", "Graph", "기업 위치 지도", "RAG", "Architecture"]
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "clean"
AURA_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
FONT_WEIGHTS = {400: "GangwonEduAll-Light", 700: "GangwonEduAll-Bold"}


def get_aura_config() -> tuple[str, str, str, str]:
    """Aura 접속 정보를 로컬 .env에서 읽는다. 값 자체는 UI에 노출하지 않는다."""
    load_dotenv(AURA_ENV_PATH)
    uri = os.getenv("AURA_URI")
    user = os.getenv("AURA_USER")
    password = os.getenv("AURA_PASSWORD")
    database = os.getenv("AURA_DATABASE")
    if not all((uri, user, password, database)):
        raise RuntimeError("AURA_URI, AURA_USER, AURA_PASSWORD, AURA_DATABASE 설정이 필요합니다.")
    return uri, user, password, database


@st.cache_resource(show_spinner=False)
def get_aura_driver(uri: str, user: str, password: str):
    """재실행마다 새 연결을 만들지 않도록 Aura 드라이버를 재사용한다."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def static_serving_enabled() -> bool:
    """정적 서빙이 켜져 있고 폰트 파일이 실제로 있는지 확인합니다."""
    try:
        enabled = bool(st.get_option("server.enableStaticServing"))
    except Exception:
        enabled = False
    return enabled and all((STATIC_DIR / f"{stem}.woff2").is_file() for stem in FONT_WEIGHTS.values())


@st.cache_data(show_spinner=False)
def _font_data_uri_cached(stem: str, fingerprint: tuple[int, int]) -> str:
    """fingerprint(수정시각, 크기)를 캐시 키에 포함해 폰트 파일 교체를 반영합니다."""
    encoded = base64.b64encode((STATIC_DIR / f"{stem}.woff2").read_bytes()).decode("ascii")
    return f"data:font/woff2;base64,{encoded}"


def font_data_uri(stem: str) -> str:
    path = STATIC_DIR / f"{stem}.woff2"
    if not path.is_file():
        return ""
    stat = path.stat()
    return _font_data_uri_cached(stem, (int(stat.st_mtime), stat.st_size))


def font_face_css(url_prefix: str, weights: tuple[int, ...] = (400, 700)) -> str:
    """정적 서빙이 가능하면 URL로, 아니면 base64로 강원교육모두를 심습니다."""
    use_url = static_serving_enabled()
    blocks = []
    for weight in weights:
        stem = FONT_WEIGHTS[weight]
        if use_url:
            source = f"url('{url_prefix}{stem}.woff2') format('woff2')"
        else:
            data_uri = font_data_uri(stem)
            if not data_uri:
                continue
            source = f"url({data_uri}) format('woff2')"
        blocks.append(
            "@font-face {"
            "font-family: 'GangwonEduAll';"
            f"src: {source};"
            f"font-weight: {weight};"
            "font-style: normal;"
            "font-display: swap;"
            "}"
        )
    return "\n".join(blocks)



@st.cache_data(show_spinner=False)
def _read_clean(name: str) -> pd.DataFrame:
    """data/clean 의 CSV 를 읽고 BOM 이 붙은 컬럼명을 정리합니다."""
    df = pd.read_csv(DATA_DIR / name, dtype=str)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_dataset_stats() -> dict[str, int]:
    """데이터셋에 실제로 들어 있는 기업 수를 셉니다."""
    try:
        overview = _read_clean("최종_기업개요.csv")
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("최종_종속기업_정리.csv")
    except FileNotFoundError:
        return {}

    parent_names = set(overview["corpNm"].dropna())
    affiliate_names = set(affiliates["afilCmpyNm"].dropna())
    subsidiary_names = set(subsidiaries["sbrdEnpNm"].dropna())
    return {
        "total": len(parent_names | affiliate_names | subsidiary_names),
        "parents": len(parent_names),
        "affiliates": len(affiliate_names),
        "subsidiaries": len(subsidiary_names),
        "relations": len(affiliates) + len(subsidiaries),
    }


@st.cache_data(show_spinner=False)
def load_top_connected(limit: int = 5) -> list[dict[str, object]]:
    """계열사 + 종속기업 수가 많은 순으로 모기업을 정렬합니다."""
    try:
        overview = _read_clean("최종_기업개요.csv")
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("최종_종속기업_정리.csv")
        merged = _read_clean("최종_모기업_계열사_종속기업_통합.csv")
    except FileNotFoundError:
        return []

    names: dict[str, str] = {}
    for frame, key, value in (
        (merged, "affiliate_crno", "affiliate_corpNm"),
        (affiliates, "afilCmpyCrno", "afilCmpyNm"),
        (merged, "top_crno", "top_corpNm"),
        (overview, "crno", "corpNm"),
    ):
        if key in frame.columns and value in frame.columns:
            names.update(frame.dropna(subset=[key, value]).set_index(key)[value].to_dict())

    industries = (
        merged.dropna(subset=["top_crno"]).drop_duplicates("top_crno").set_index("top_crno")["top_sicNm"].to_dict()
        if "top_sicNm" in merged.columns
        else {}
    )

    affiliate_count = affiliates.groupby("crno")["afilCmpyNm"].nunique()
    subsidiary_count = subsidiaries.groupby("crno")["sbrdEnpNm"].nunique()
    total = affiliate_count.add(subsidiary_count, fill_value=0).sort_values(ascending=False)

    rows: list[dict[str, object]] = []
    for crno, count in total.items():
        name = names.get(crno)
        if not name:  # 이름을 확인할 수 없는 법인등록번호는 건너뜁니다.
            continue
        industry = industries.get(crno)
        rows.append(
            {
                "name": name,
                "industry": industry if isinstance(industry, str) else "",
                "affiliates": int(affiliate_count.get(crno, 0)),
                "subsidiaries": int(subsidiary_count.get(crno, 0)),
                "total": int(count),
            }
        )
        if len(rows) == limit:
            break
    return rows


def get_site_content() -> dict[str, object]:
    return {
        "service_name": "B2B Knowledge Graph Explorer",
        "wordmark": "기업 연결 관계",
        "topbar": "공시 데이터부터 지식그래프까지, 기업 간의 연결 관계를 한 화면에서 탐색하세요",
        "hero_lines": ["기업 간의 연결 관계를 지식그래프로 구축한 프로젝트"],
        "statement": ["계열 관계", "종속 구조", "공급망 연결"],
        "tagline": "기업 관계를 지식 그래프로 연결해 산업 생태계와 잠재 고객사를 빠르게 탐색하는 B2B 리서치 서비스",
        "summary": "영업·마케팅 담당자가 기업명, 산업, 키워드, 관계 유형을 한 화면에서 탐색하고 그래프 기반 질의응답으로 다음 접점을 발견하도록 돕습니다.",
        "core_questions": [
            {"question": "특정 기업과 연결된 계열사는 어디인가?", "feature": "지식그래프 탐색"},
            {"question": "이 기업과 유사한 일을 하는 기업은 어디인가?", "feature": "유사 기업 추천"},
            {"question": "특정 산업에서 영업 대상이 될 만한 기업은 어디인가?", "feature": "산업별 기업 탐색 및 필터링"},
            {"question": "해당 기업의 규모와 최근 상황은 어떤가?", "feature": "그래프 RAG"},
        ],
        "stack": ["pandas", "OpenAI Embeddings", "OpenAI LLM", "Neo4j", "Cypher", "Streamlit"],
    }


@st.cache_data(show_spinner=False)
def load_company_options() -> list[dict[str, object]]:
    """그래프에서 선택할 수 있는 모기업 목록을 연결 수가 많은 순으로 만듭니다."""
    try:
        overview = _read_clean("최종_기업개요.csv")
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("최종_종속기업_정리.csv")
        merged = _read_clean("최종_모기업_계열사_종속기업_통합.csv")
    except FileNotFoundError:
        return []

    names: dict[str, str] = {}
    for frame, key, value in (
        (merged, "top_crno", "top_corpNm"),
        (overview, "crno", "corpNm"),
    ):
        if key in frame.columns and value in frame.columns:
            names.update(frame.dropna(subset=[key, value]).set_index(key)[value].to_dict())

    affiliate_count = affiliates.groupby("crno")["afilCmpyNm"].nunique()
    subsidiary_count = subsidiaries.groupby("crno")["sbrdEnpNm"].nunique()
    total = affiliate_count.add(subsidiary_count, fill_value=0).sort_values(ascending=False)

    options: list[dict[str, object]] = []
    for crno, count in total.items():
        name = names.get(crno)
        if not name:
            continue
        options.append(
            {
                "crno": crno,
                "name": name,
                "affiliates": int(affiliate_count.get(crno, 0)),
                "subsidiaries": int(subsidiary_count.get(crno, 0)),
                "total": int(count),
            }
        )
    return options


@st.cache_data(show_spinner=False)
def load_company_relations(crno: str) -> list[dict[str, str]]:
    """선택한 기업의 계열사·종속기업을 관계 유형과 함께 돌려줍니다."""
    try:
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("최종_종속기업_정리.csv")
    except FileNotFoundError:
        return []

    rows: list[dict[str, str]] = []
    for name in affiliates.loc[affiliates["crno"] == crno, "afilCmpyNm"].dropna().unique():
        rows.append({"name": str(name), "relation": "계열사"})
    for name in subsidiaries.loc[subsidiaries["crno"] == crno, "sbrdEnpNm"].dropna().unique():
        rows.append({"name": str(name), "relation": "종속기업"})
    return rows


def _representative_text(value: object) -> str:
    """대표자 배열 문자열을 화면 표시용 텍스트로 변환합니다."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text.strip("[]\\\"'")
    if isinstance(parsed, list):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    return str(parsed)


@st.cache_data(show_spinner=False)
def load_company_map_data() -> pd.DataFrame:
    """카카오 지오코딩이 완료된 모기업·종속기업 위치를 결합합니다."""
    parent_geocode_path = DATA_DIR / "parent_company_geocodes.csv"
    subsidiary_geocode_path = DATA_DIR / "subsidiary_company_geocodes.csv"
    overview_path = DATA_DIR / "최종_기업개요.csv"
    if not parent_geocode_path.is_file():
        return pd.DataFrame()

    geocodes = [
        pd.read_csv(parent_geocode_path, dtype=str, encoding="utf-8-sig").assign(entity_type="모기업")
    ]
    if subsidiary_geocode_path.is_file():
        geocodes.append(
            pd.read_csv(subsidiary_geocode_path, dtype=str, encoding="utf-8-sig").assign(entity_type="종속기업")
        )
    merged = pd.concat(geocodes, ignore_index=True)

    if not overview_path.is_file():
        overview = pd.DataFrame(columns=["crno", "enpRprFnm", "enpTlno", "region", "sicNm"])
    else:
        overview = _read_clean("최종_기업개요.csv")
    overview_columns = [
        column
        for column in ("crno", "enpRprFnm", "enpTlno", "enpBsadr", "region", "sicNm")
        if column in overview.columns
    ]
    merged = merged.merge(
        overview[overview_columns].drop_duplicates("crno"),
        on="crno",
        how="left",
        suffixes=("", "_overview"),
    )
    merged["latitude"] = pd.to_numeric(merged["latitude"], errors="coerce")
    merged["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce")
    merged = merged[
        (merged["status"] == "ok")
        & merged["latitude"].notna()
        & merged["longitude"].notna()
    ].copy()
    representative_values = (
        merged["enpRprFnm"]
        if "enpRprFnm" in merged.columns
        else pd.Series("", index=merged.index)
    )
    phone_values = (
        merged["enpTlno"]
        if "enpTlno" in merged.columns
        else pd.Series("", index=merged.index)
    )
    merged["representative"] = representative_values.map(_representative_text)
    merged["phone"] = phone_values.fillna("").astype(str).replace("nan", "")
    merged["display_address"] = merged["address"].fillna("").astype(str)
    region_values = merged["region"] if "region" in merged.columns else pd.Series("", index=merged.index)
    sic_values = merged["sicNm"] if "sicNm" in merged.columns else pd.Series("", index=merged.index)
    merged["region"] = region_values.fillna("").astype(str).replace("nan", "")
    merged["sicNm"] = sic_values.fillna("").astype(str).replace("nan", "")
    return merged


def build_company_graph(
    center: str,
    relations: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """중심 기업을 가운데 두고 연결 기업을 원형으로 배치합니다."""
    nodes: list[dict[str, object]] = [
        {"id": center, "group": "핵심기업", "size": 40, "x": 480, "y": 320, "description": ""}
    ]
    edges: list[dict[str, object]] = []

    count = len(relations)
    # 12개를 넘으면 안쪽/바깥쪽 두 겹으로 나눠 라벨이 겹치지 않게 합니다.
    two_rings = count > 12
    for index, row in enumerate(relations):
        if two_rings:
            radius = 175 if index % 2 == 0 else 268
        else:
            radius = 215
        angle = -math.pi / 2 + 2 * math.pi * index / max(count, 1)
        nodes.append(
            {
                "id": row["name"],
                "group": row["relation"],
                "size": 16 if two_rings else 22,
                "x": 480 + radius * math.cos(angle) * 1.42,
                "y": 320 + radius * math.sin(angle),
                "description": "",
            }
        )
        edges.append({"source": center, "target": row["name"], "relation": row["relation"], "confidence": 1.0})
    return nodes, edges


def build_architecture_steps() -> list[dict[str, str]]:
    return [
        {"step": "01", "title": "데이터 수집", "body": "금융위원회 기업기본정보 API와 웹 문서를 수집합니다."},
        {"step": "02", "title": "전처리", "body": "중복, 누락, 짧은 문서를 정제하고 기업 식별자를 통합합니다."},
        {"step": "03", "title": "LLM 트리플 추출", "body": "기업, 산업, 제품, 관계를 subject-predicate-object 구조로 추출합니다."},
        {"step": "04", "title": "Neo4j 통합", "body": "Cypher로 노드와 엣지를 적재해 지식그래프를 구축합니다."},
        {"step": "05", "title": "품질 평가", "body": "관계 유형, confidence, 근거 문장을 기준으로 그래프 품질을 점검합니다."},
        {"step": "06", "title": "Text2Cypher + RAG", "body": "자연어 질문을 그래프 질의와 문서 근거 답변으로 변환합니다."},
        {"step": "07", "title": "Streamlit 서비스", "body": "검색, 상세, 그래프, 추천, 질의응답 화면을 제공합니다."},
    ]


def css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

    {font_face_css("app/static/")}

    :root {{
        --ink: #101010;
        --mute: #8c887f;
        --paper: #fbfaf8;
        --line: #e2dfd8;
        --line-strong: #101010;
        --accent: #b4aaa1;
        --display: 'GangwonEduAll', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
        --sans: 'GangwonEduAll', 'Inter', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
        --label: 'Inter', 'GangwonEduAll', sans-serif;
    }}

    .stApp {{ background: var(--paper); color: var(--ink); }}
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div {{ font-family: var(--sans); }}
    /* 아이콘은 Material Symbols 를 그대로 써야 한다. 위 전역 규칙에서 되돌린다. */
    [data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important; font-style: normal !important;
        letter-spacing: normal !important; text-transform: none !important;
        white-space: nowrap !important; word-break: normal !important; direction: ltr;
        font-feature-settings: 'liga'; -webkit-font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ max-width: 1240px; padding-top: 1.6rem; padding-bottom: 5rem; }}

    h1, h2, h3, h4 {{
        font-family: var(--display) !important;
        font-weight: 400 !important;
        letter-spacing: -0.015em !important;
        color: var(--ink);
        word-break: keep-all;
        overflow-wrap: break-word;
    }}
    h2 {{ font-size: 2.6rem !important; line-height: 1.12 !important; margin: .2rem 0 1.4rem !important; }}
    h3 {{ font-size: 1.45rem !important; }}

    /* 상단 밴드: 브랜드 + 안내 문구 (화면 전체 폭) */
    .topbar {{
        background: var(--accent); color: var(--ink);
        margin: -1.6rem -100vw 0; padding: 1.15rem 100vw;
    }}
    .topbar-inner {{
        display:flex; align-items:center; gap: 1.4rem; flex-wrap: wrap;
    }}
    .topbar-inner .mark {{
        font-family: var(--display); font-size: 1.45rem; line-height: 1.2; font-weight: 700;
        letter-spacing: -0.01em; white-space: nowrap;
    }}
    .topbar-msg {{ flex: 0 1 auto; max-width: 46%; font-size: .9rem; }}

    /* 내비게이션(라디오)을 탭처럼 보이게: 밴드 위로 끌어올려 오른쪽 배치 */
    /* 라디오를 감싼 컨테이너가 내용 폭에 맞춰 줄어들어 우측 정렬이 먹지 않으므로 넓힌다. */
    [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {{ width: 100% !important; }}
    /* streamlit 헤더(z-index 999990)가 상단 60px 를 덮어 내비 클릭을 가로채므로,
       빈 영역은 클릭이 통과하게 하고 내비를 그 위로 올린다. */
    [data-testid="stHeader"] {{ pointer-events: none !important; background: transparent !important; }}
    [data-testid="stHeader"] button, [data-testid="stHeader"] a,
    [data-testid="stToolbar"] > *, [data-testid="stMainMenu"] {{ pointer-events: auto !important; }}

    [data-testid="stRadio"] {{
        width: 100%; margin-top: -3.4rem; margin-bottom: 3.2rem; position: relative; z-index: 999991;
    }}
    [data-testid="stRadio"] > label[data-testid="stWidgetLabel"] {{ display: none !important; }}
    [data-testid="stRadio"] [role="radiogroup"] {{
        width: 100%; display: flex; justify-content: flex-end; gap: 2.4rem; flex-wrap: wrap;
    }}
    [data-testid="stRadioOption"] {{
        margin: 0 !important; padding: 0 !important; background: transparent !important; cursor: pointer;
    }}
    /* streamlit 의 안내 팝업이 밴드 우측(내비 영역)을 덮어 클릭을 가로채므로 숨긴다. */
    [data-testid="stSkillsNudgeAnchor"], [data-testid="stSkillsNudge"] {{
        display: none !important; pointer-events: none !important;
    }}

    /* 라디오 동그라미 제거: 라벨 텍스트(마크다운 컨테이너)만 남긴다. */
    [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) {{
        display: none !important;
    }}
    [data-testid="stRadioOption"] p {{
        font-family: var(--label) !important; font-size: .76rem !important; letter-spacing: .18em !important;
        text-transform: uppercase; color: rgba(16, 16, 16, .58) !important; margin: 0 !important;
        padding-bottom: .3rem; border-bottom: 1px solid transparent;
    }}
    [data-testid="stRadioOption"]:hover p {{ color: var(--ink) !important; }}
    [data-testid="stRadioOption"][data-selected="true"] p {{
        color: var(--ink) !important; border-bottom-color: var(--ink);
    }}

    .rule {{ height:1px; background: var(--line-strong); margin: 0 0 3.4rem; }}
    .rule.soft {{ background: var(--line); margin: 2.6rem 0; }}

    /* 홈 3단 패널 */
    .panel {{ padding: 2.2rem 1.8rem 2rem 0; border-right: 1px solid var(--line); min-height: 320px; }}
    .panel.panel-last {{ border-right: 0; padding-right: 0; }}
    .panel-title {{ font-size: 1.28rem !important; margin: 0 0 1.5rem !important; }}
    .panel-foot {{ margin-top: 1.4rem; font-family: var(--label); font-size: .68rem; letter-spacing: .06em; color: var(--mute); line-height: 1.7; }}
    .stat {{ font-size: 1.22rem; line-height: 1.65; margin: 0 0 1.6rem; }}
    .stat b {{ display:block; font-family: var(--display); font-size: 3.6rem; font-weight: 400; line-height: 1; color: var(--ink); }}
    .stat-rows div {{
        display:flex; justify-content:space-between; align-items: baseline;
        padding: .55rem 0; border-top: 1px solid var(--line);
    }}
    .stat-rows span {{ font-family: var(--label); font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--mute); }}
    .stat-rows b {{ font-weight: 400; font-size: 1.02rem; }}

    /* 연결이 많은 기업 랭킹 */
    .ranking {{ list-style: none; margin: 0; padding: 0; }}
    .ranking li {{
        display:grid; grid-template-columns: 22px 1fr 3.2rem; align-items: baseline; gap: .6rem;
        padding: .72rem 0; border-top: 1px solid var(--line);
    }}
    .ranking li:first-child {{ border-top: 0; }}
    .ranking .rank {{ font-family: var(--label); font-size: .8rem; color: var(--accent); }}
    .ranking .rank-body b {{ display:block; font-weight: 400; font-size: .98rem; word-break: keep-all; line-height: 1.4; }}
    .ranking .rank-body small {{ color: var(--mute); font-size: .78rem; }}
    .ranking .rank-total {{ font-family: var(--display); font-size: 1.25rem; color: var(--ink); text-align: right; }}

    /* editorial rows */
    .kicker {{ font-family: var(--label); font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--mute); display:block; margin-bottom: 1.1rem; }}
    .row {{ display:grid; grid-template-columns: 78px 1.05fr 1.6fr 96px; gap: 1.6rem; align-items: baseline; padding: 1.35rem 0; border-top: 1px solid var(--line); }}
    .row:last-child {{ border-bottom: 1px solid var(--line); }}
    .row .idx {{ font-family: var(--label); font-size: .74rem; letter-spacing: .12em; color: var(--mute); }}
    .row .title {{ font-family: var(--display); font-size: 1.5rem; letter-spacing: -0.01em; }}
    .row .body {{ color: var(--mute); font-size: .93rem; line-height: 1.8; font-weight: 400; }}
    .row .tail {{ font-family: var(--label); text-align:right; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--mute); }}
    .row:hover .title {{ font-weight: 700; }}

    /* question list */
    .q {{ padding: 1.1rem 0 .1rem; border-top: 1px solid var(--line); }}
    .q small {{ font-family: var(--label); display:block; color: var(--mute); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }}
    /* 질문 버튼: 눌러서 RAG 로 넘어가는 링크처럼 보이게 */
    [data-testid="stVerticalBlock"] .stButton button[kind="secondary"] {{
        background: transparent !important; border: 0 !important; border-radius: 0 !important;
        padding: .1rem 0 1.1rem !important; text-align: left !important; justify-content: flex-start !important;
        text-transform: none !important; letter-spacing: 0 !important;
    }}
    [data-testid="stVerticalBlock"] .stButton button[kind="secondary"] p {{
        font-family: var(--sans) !important; font-size: 1.02rem !important; font-weight: 700 !important;
        color: var(--ink) !important; text-align: left !important;
    }}
    [data-testid="stVerticalBlock"] .stButton button[kind="secondary"]:hover p {{ color: var(--accent) !important; }}

    .lede {{ font-size: 1.12rem; line-height: 1.9; font-weight: 400; color: var(--ink); }}
    .note {{ color: var(--mute); font-size: .88rem; line-height: 1.85; font-weight: 400; border-left: 1px solid var(--line-strong); padding-left: 1rem; margin-top: 1.6rem; }}

    /* answer */
    .answer {{ border-top: 1px solid var(--line-strong); padding-top: 1.6rem; margin-top: 1.8rem; }}
    .answer .label {{ font-family: var(--label); font-size: .7rem; letter-spacing: .2em; text-transform: uppercase; color: var(--mute); display:block; margin-bottom: .55rem; }}
    .answer p {{ font-size: 1rem; line-height: 1.9; font-weight: 400; margin: 0 0 1.6rem; }}
    .answer code {{ font-family: 'SFMono-Regular', Consolas, monospace; display:block; background: #f1efea; color: var(--ink); padding: .9rem 1rem; font-size: .82rem; line-height: 1.6; white-space: pre-wrap; }}

    /* stack */
    .stack {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }}
    .stack span {{ font-family: var(--label); border: 1px solid var(--line); border-radius: 999px; padding: .42rem .95rem; font-size: .78rem; letter-spacing: .04em; color: var(--mute); }}

    /* legend */
    .legend {{ font-family: var(--label); display:flex; flex-wrap:wrap; gap:1.6rem; margin: 1.2rem 0 .4rem; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--mute); }}
    .legend i {{ display:inline-block; width:9px; height:9px; border-radius:50%; border:1px solid var(--ink); margin-right:.5rem; }}

    /* streamlit widgets */
    /* 세그먼트 컨트롤: 비활성 회색, 활성 흰 배경 + 검정 글씨 */
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{
        background: transparent !important;
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: .45rem 1.1rem !important;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"] * {{
        color: var(--mute) !important; font-size: .82rem !important; letter-spacing: .02em;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover * {{ color: var(--ink) !important; }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][data-selected="true"],
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {{
        background: #ffffff !important;
        border-color: var(--ink) !important;
    }}
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][data-selected="true"] *,
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] * {{
        color: var(--ink) !important;
    }}

    /* 탭: 비활성 회색, 활성 검정 (streamlit 버전별 마크업 모두 커버) */
    .stTabs [role="tablist"], [data-baseweb="tab-list"] {{
        gap: 2.4rem !important; background: transparent !important; border-bottom: 0 !important;
    }}
    .stTabs [role="tab"], button[data-baseweb="tab"] {{
        background: transparent !important; padding: 0 0 .85rem 0 !important;
    }}
    .stTabs [role="tab"] p, button[data-baseweb="tab"] p {{
        font-family: var(--label) !important; font-size: .76rem !important; letter-spacing: .18em !important;
        text-transform: uppercase; color: rgba(16, 16, 16, .58) !important;
    }}
    .stTabs [role="tab"]:hover p {{ color: var(--ink) !important; }}
    .stTabs [role="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] p {{ color: var(--ink) !important; }}
    .stTabs .react-aria-SelectionIndicator, [data-baseweb="tab-highlight"] {{
        background: var(--ink) !important; background-color: var(--ink) !important; height: 1px !important;
    }}
    [data-baseweb="tab-border"] {{ display:none !important; }}

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background: transparent !important; border: 0 !important; border-bottom: 1px solid var(--line-strong) !important;
        border-radius: 0 !important; font-size: 1rem !important; padding-left: 0 !important;
    }}
    .stTextInput label, .stSelectbox label, [data-testid="stWidgetLabel"] label {{
        font-family: var(--label) !important; font-size: .7rem !important; letter-spacing: .2em !important; text-transform: uppercase; color: var(--mute) !important;
    }}
    .stButton button {{
        background: var(--accent) !important; border: 1px solid var(--accent) !important; border-radius: 999px;
        padding: .6rem 1.7rem; font-family: var(--label); font-size: .76rem; letter-spacing: .16em; text-transform: uppercase;
        box-shadow: none !important;
    }}
    .stButton button, .stButton button * {{ color: var(--ink) !important; }}
    .stButton button:hover {{ background: transparent !important; border-color: var(--ink) !important; }}

    /* 포커스 링: 기본 빨강 대신 대표 색상 */
    *:focus, *:focus-visible {{ outline-color: var(--accent) !important; }}
    *:focus-visible, .stApp *:focus {{
        box-shadow: 0 0 0 3px rgba(180, 170, 161, .45) !important;
    }}

    /* 입력부: 다크 테마 잔재 제거 */
    [data-testid="stTextInputRootElement"], .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox .react-aria-ComboBox > div {{
        background: #ffffff !important; border: 0 !important; border-bottom: 1px solid var(--line-strong) !important;
        border-radius: 0 !important; box-shadow: none !important;
    }}
    [data-testid="stTextInputField"], .stSelectbox input, .stSelectbox [data-baseweb="select"] * {{
        color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    }}
    .stCaption p {{ color: var(--mute) !important; font-size: .8rem !important; }}

    [data-testid="stSidebar"] {{ background: var(--paper); border-right: 1px solid var(--line); }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
        font-family: var(--sans); color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] .sb-mark {{
        font-family: var(--display); font-size: 1.5rem; line-height: 1.25; font-weight: 700; word-break: keep-all;
    }}
    [data-testid="stSidebar"] .sb-label {{
        font-family: var(--label); font-size: .68rem; letter-spacing: .2em; text-transform: uppercase;
        display:block; margin-top: 1.6rem; font-weight: 700;
    }}
    [data-testid="stSidebar"] .sb-value {{ font-size: .95rem; margin-top: .3rem; line-height: 1.6; }}
    [data-testid="stSidebar"] [data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}

    [data-testid="stDataFrame"] {{ border: 1px solid var(--line); }}

    /* RAG 챗봇: 다크 테마 잔재 제거 */
    [data-testid="stChatMessage"] {{
        background: #ffffff !important; border: 1px solid var(--line) !important; border-radius: 0 !important;
    }}
    [data-testid="stChatMessage"] *, [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{
        color: var(--ink) !important;
    }}
    [data-testid="stChatInput"], .stChatInput > div {{
        background: #ffffff !important; border: 1px solid var(--line-strong) !important; border-radius: 0 !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; background: transparent !important;
    }}
    [data-testid="stExpander"] {{ background: transparent !important; border: 1px solid var(--line) !important; border-radius: 0 !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * , details summary * {{
        color: var(--ink) !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {{ color: var(--ink) !important; }}
    /* 채팅 입력이 놓이는 하단 영역: 기본 다크 배경 -> 대표 색상 */
    [data-testid="stBottom"] > div {{ background: var(--accent) !important; }}
    /* 입력창을 본문(채팅 말풍선)과 같은 폭으로 맞춘다. 갈색 배경 띠는 전체 폭 유지. */
    [data-testid="stBottomBlockContainer"] {{
        background: transparent !important;
        max-width: 1240px !important; margin-left: auto !important; margin-right: auto !important;
    }}

    /* 답변 속 인라인 코드가 검은 블록으로 보이지 않도록 */
    [data-testid="stChatMessage"] code, .stMarkdown code, [data-testid="stExpander"] code {{
        background: #ece8e2 !important; color: var(--ink) !important; padding: .05rem .3rem;
    }}
    [data-testid="stChatMessage"] pre, .stMarkdown pre, [data-testid="stCodeBlock"] {{
        background: #f1efea !important;
    }}
    [data-testid="stChatMessage"] pre *, .stMarkdown pre *, [data-testid="stCodeBlock"] * {{
        color: var(--ink) !important;
    }}

    /* 답변 대기 캐릭터: 1초에 한 번씩 시계방향 90도 (4초에 한 바퀴) */
    @keyframes bot-thinking-tick {{
        0%, 20% {{ transform: rotate(0deg); }}
        25%, 45% {{ transform: rotate(90deg); }}
        50%, 70% {{ transform: rotate(180deg); }}
        75%, 95% {{ transform: rotate(270deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    img[alt="조회 중"] {{
        animation: bot-thinking-tick 4s ease-in-out infinite;
        transform-origin: 50% 50%;
    }}

    /* 이미지 아바타(캐릭터)는 배경 없이 그림만 보이게 */
    [data-testid="stChatMessage"] > img {{
        background: transparent !important; object-fit: contain; border-radius: 0 !important;
        width: 4rem !important; height: 4rem !important; flex-shrink: 0;  /* 기본 2rem 의 2배 */
    }}
    [data-testid="stChatMessageAvatarUser"] {{ background: var(--accent) !important; }}
    [data-testid="stChatMessageAvatarAssistant"] {{ background: var(--ink) !important; }}
    [data-testid="stChatMessageAvatarUser"] *, [data-testid="stChatMessageAvatarAssistant"] * {{
        color: var(--paper) !important;
    }}

    @media (max-width: 1100px) {{
        .topbar-msg {{ flex-basis: 100%; }}
        .panel {{ border-right: 0; min-height: 0; padding: 1.6rem 0; border-bottom: 1px solid var(--line); }}
    }}

    @media (max-width: 900px) {{
        .row {{ grid-template-columns: 1fr; gap: .5rem; }}
        .row .tail {{ text-align:left; }}
    }}
    </style>
    """


def render_graph_svg(nodes: list[dict[str, object]], edges: list[dict[str, object]]) -> str:
    lookup = {node["id"]: node for node in nodes}
    color_by_relation = {"계열사": "#b4aaa1", "종속기업": "#cfc7bf"}
    height = 680

    edge_markup: list[str] = []
    for edge in edges:
        source, target = lookup[edge["source"]], lookup[edge["target"]]
        sx, sy, tx, ty = source["x"], source["y"], target["x"], target["y"]
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1
        cx, cy = mx - dy / length * 34, my + dx / length * 34
        stroke = color_by_relation.get(str(edge["relation"]), "#cfccc4")
        edge_markup.append(
            f'<path d="M {sx:.1f} {sy:.1f} Q {cx:.1f} {cy:.1f} {tx:.1f} {ty:.1f}" fill="none" stroke="{stroke}" stroke-width="1" />'
        )

    node_markup: list[str] = []
    for node in nodes:
        is_core = node["group"] == "핵심기업"
        fill = "#b4aaa1" if is_core else color_by_relation.get(str(node["group"]), "#dbd6d1")
        radius = int(node["size"])
        label = str(node["id"])
        if len(label) > 14:  # 긴 상호는 줄여서 라벨끼리 겹치지 않게 합니다.
            label = label[:13] + "…"
        node_markup.append(
            f'<circle cx="{node["x"]:.1f}" cy="{node["y"]:.1f}" r="{radius}" fill="{fill}" />'
            f'<title>{html.escape(str(node["id"]))}</title>'
            f'<text x="{node["x"]:.1f}" y="{node["y"] + radius + 20:.1f}" text-anchor="middle" '
            f'font-size="{15 if is_core else 12}" fill="#101010" font-family="GangwonEduAll, sans-serif">{html.escape(label)}</text>'
        )

    return f"""
    <style>
    {font_face_css("/app/static/", weights=(400,))}
    body {{ margin: 0; }}
    </style>
    <div style="background:#fbfaf8;border-top:1px solid #101010;border-bottom:1px solid #e2dfd8;">
      <svg viewBox="0 0 960 {height}" width="100%" height="{height}" role="img" aria-label="기업 관계 지식 그래프">
        <rect x="0" y="0" width="960" height="{height}" fill="#fbfaf8" />
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """


def render_aura_graph_html(nodes: list[dict[str, object]], edges: list[dict[str, object]]) -> str:
    """Aura 그래프를 드래그·줌 가능한 네트워크로 렌더링한다."""
    node_colors = {
        "ParentCompany": "#b4aaa1",
        "SubsidiaryCompany": "#cfc7bf",
        "Section": "#d7b98f",
        "Region": "#b8c5c8",
        "News": "#d6d1ca",
    }
    edge_colors = {
        "AFFILIATED_WITH": "#b4aaa1",
        "HAS_SUBSIDIARY": "#cfc7bf",
        "IN_INDUSTRY": "#d7b98f",
        "LOCATED_IN": "#b8c5c8",
        "RELATED_TO": "#d6d1ca",
    }
    network = Network(height="680px", width="100%", directed=True, bgcolor="#fbfaf8", font_color="#101010", cdn_resources="in_line")
    network.set_options(
        """
        var options = {
          "interaction": {"dragNodes": true, "dragView": true, "zoomView": true},
          "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -18000, "springLength": 145}},
          "edges": {"smooth": {"type": "continuous"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.45}}}
        }
        """
    )
    for node in nodes:
        is_center = node["level"] == 0
        network.add_node(
            node["id"],
            label=str(node["label"]),
            title=html.escape(f"{node['label']} · {node['group']} · {node['level']}단계"),
            color="#b4aaa1" if is_center else node_colors.get(str(node["group"]), "#dbd6d1"),
            font={"color": "#101010", "size": 18 if is_center else 13},
            size=30 if is_center else 17,
        )
    for edge in edges:
        relation = str(edge["relation"])
        network.add_edge(
            edge["source"],
            edge["target"],
            title=relation,
            color=edge_colors.get(relation, "#cfccc4"),
            width=1.2,
        )
    return network.generate_html()


def compact(markup: str) -> str:
    """줄바꿈/들여쓰기를 제거합니다. 빈 줄이 있으면 streamlit 의 마크다운 파서가 HTML 블록을 끊습니다."""
    return "".join(line.strip() for line in markup.splitlines())


def render_topbar(content: dict[str, object]) -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        compact(f"""
        <div class="topbar">
            <div class="topbar-inner">
                <div class="mark">{content['wordmark']}</div>
                <span class="topbar-msg">{content['topbar']}</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_hero_band(content: dict[str, object]) -> None:
    """홈 본문: 데이터 규모와 연결 관계가 많은 기업."""
    stats = load_dataset_stats()
    ranking = load_top_connected(5)

    left, middle = st.columns([1, 1.2], gap="large")

    with left:
        if stats:
            body = f"""
            <div class="panel">
                <h3 class="panel-title">지식그래프에는 지금,</h3>
                <p class="stat"><b>{stats['total']:,}</b>개의<br>기업이 연결되어 있습니다.</p>
                <div class="stat-rows">
                    <div><span>모기업</span><b>{stats['parents']:,}</b></div>
                    <div><span>계열사</span><b>{stats['affiliates']:,}</b></div>
                    <div><span>종속기업</span><b>{stats['subsidiaries']:,}</b></div>
                    <div><span>관계 수</span><b>{stats['relations']:,}</b></div>
                </div>
                <div class="panel-foot">금융위원회 기업기본정보 · 계열회사 · 연결대상종속기업</div>
            </div>
            """
        else:
            body = '<div class="panel"><h3 class="panel-title">데이터셋을 찾을 수 없습니다</h3><p class="panel-foot">data/clean 경로를 확인해 주세요.</p></div>'
        st.markdown(compact(body), unsafe_allow_html=True)

    with middle:
        if ranking:
            items = "".join(
                f"""
                <li>
                    <span class="rank">{index}.</span>
                    <span class="rank-body">
                        <b>{html.escape(row['name'])}</b>
                        <small>계열사 {row['affiliates']} · 종속기업 {row['subsidiaries']}</small>
                    </span>
                    <span class="rank-total">{row['total']}</span>
                </li>
                """
                for index, row in enumerate(ranking, start=1)
            )
            body = f"""
            <div class="panel panel-last">
                <h3 class="panel-title">연결 관계가 가장 많은 기업</h3>
                <ol class="ranking">{items}</ol>
                <div class="panel-foot">계열사 + 종속기업 수 기준 · 기업명이 확인되는 모기업만 집계</div>
            </div>
            """
        else:
            body = '<div class="panel panel-last"><h3 class="panel-title">집계할 관계 데이터가 없습니다</h3></div>'
        st.markdown(compact(body), unsafe_allow_html=True)


def ask_in_rag(question: str) -> None:
    """질문을 RAG 챗봇으로 넘기고 해당 섹션으로 이동한다.

    on_click 콜백에서만 호출한다. 스크립트 본문에서 위젯 키(nav)를 바꾸면
    streamlit 이 '위젯 생성 후 수정' 으로 막는다.
    """
    st.session_state["rag_pending"] = question
    st.session_state["nav"] = "RAG"


def render_problem_and_questions(content: dict[str, object]) -> None:
    st.markdown("## 흩어진 기업 정보를<br>연결 가능한 영업 지도로", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(f'<p class="lede">{content["summary"]}</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="note"><b>현재</b> — 포털 검색, 기업 홈페이지, 뉴스, 공시, 엑셀 리스트를 사람이 직접 비교하고 관계를 추론합니다.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note"><b>해결</b> — 기업, 산업, 제품, 거래·협력 관계를 그래프로 연결하고 자연어 질문을 그래프 질의로 전환합니다.</div>',
            unsafe_allow_html=True,
        )
    with right:
        for index, item in enumerate(content["core_questions"]):
            st.markdown(
                f'<div class="q"><small>{item["feature"]}</small></div>',
                unsafe_allow_html=True,
            )
            st.button(
                item["question"],
                key=f"q_{index}",
                use_container_width=True,
                on_click=ask_in_rag,
                args=(str(item["question"]),),
            )


def render_graph_explorer() -> None:
    st.markdown("## 기업 관계 그래프 탐색")
    options = load_company_options()
    if not options:
        st.error("기업 데이터를 불러오지 못했습니다. data/clean 경로를 확인해 주세요.")
        return

    labels = [f"{row['name']}  ·  계열사 {row['affiliates']} · 종속기업 {row['subsidiaries']}" for row in options]
    picker, relation_col, limit_col = st.columns([1.6, 1, 1], gap="large")
    with picker:
        chosen = st.selectbox("기업 선택", labels, index=0)
    selected = options[labels.index(chosen)]

    with relation_col:
        relation = st.segmented_control("관계 유형", ["전체", "계열사", "종속기업"], default="전체") or "전체"
    with limit_col:
        limit = st.slider("표시할 연결 기업 수", min_value=4, max_value=40, value=16, step=2)

    relation_type = {"계열사": "AFFILIATED_WITH", "종속기업": "HAS_SUBSIDIARY"}.get(relation)
    try:
        uri, user, password, database = get_aura_config()
        driver = get_aura_driver(uri, user, password)
        matches = search_companies(driver, str(selected["name"]), database)
        exact_matches = [row for row in matches if row["name"] == selected["name"]]
        if not exact_matches:
            st.info("선택한 기업은 Aura 그래프에 아직 없습니다.")
            return
        paths = fetch_graph_paths(
            driver,
            str(exact_matches[0]["id"]),
            database,
            path_limit=100,
            relationship_type=relation_type,
        )
    except Exception as exc:
        st.error(f"Aura 관계 조회에 실패했습니다: {type(exc).__name__}: {exc}")
        return

    if not paths:
        st.info("선택한 관계 유형에 해당하는 Aura 그래프 연결이 없습니다.")
        return

    total = len(paths)
    shown = paths[:limit]
    center_id = str(exact_matches[0]["id"])
    nodes, edges = build_graph_elements(shown, center_id=center_id)
    st.markdown(
        f"""
        <div class="legend">
            <span><i style="background:#b4aaa1;border-color:#b4aaa1"></i>계열 관계</span>
            <span><i style="background:#cfc7bf;border-color:#cfc7bf"></i>종속 관계</span>
            <span>연결 {total}개 중 {len(shown)}개 표시</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(render_aura_graph_html(nodes, edges), height=700, scrolling=False)
    st.caption("AuraDB에서 읽어온 실제 관계입니다. 드래그·줌으로 그래프를 탐색할 수 있습니다.")


def render_company_map() -> None:
    """카카오 지오코딩이 완료된 기업 위치를 지도와 상세 카드로 표시합니다."""
    st.markdown("## 기업 위치 지도")
    st.markdown(
        '<p class="lede">카카오 주소 검색으로 변환된 모기업·종속기업 위치를 지도에서 선택하면 기업 정보를 확인할 수 있습니다.</p>',
        unsafe_allow_html=True,
    )

    map_df = load_company_map_data()
    if map_df.empty:
        st.warning("지오코딩된 기업 데이터가 없습니다. 먼저 geocode_parent_companies.py를 실행해 주세요.")
        return

    region_options = ["전체"] + sorted(str(value) for value in map_df["region"].dropna().unique() if str(value).strip())
    filter_col, search_col = st.columns([1, 2], gap="large")
    with filter_col:
        entity_options = ["전체", "모기업", "종속기업"]
        selected_entity = st.selectbox("기업 유형", entity_options, key="company_map_entity")
        selected_region = st.selectbox("지역 필터", region_options, key="company_map_region")
    with search_col:
        company_query = st.text_input(
            "기업명 검색",
            placeholder="예: 롯데쇼핑, 한화, 삼성",
            key="company_map_query",
        ).strip()
    visible = map_df if selected_entity == "전체" else map_df[map_df["entity_type"] == selected_entity]
    if selected_region != "전체":
        visible = visible[visible["region"] == selected_region]
    if company_query:
        visible = visible[
            visible["name"].astype(str).str.contains(company_query, case=False, regex=False, na=False)
        ]
    # PyDeck 선택 인덱스와 DataFrame 행 번호를 일치시킵니다.
    visible = visible.copy().reset_index(drop=True)
    visible["map_id"] = visible["id"].astype(str)

    if company_query and visible.empty:
        st.warning(f"'{company_query}'에 해당하는 기업을 찾지 못했습니다.")
        return

    result_caption = f"검색 결과 {len(visible):,}개" if company_query else f"표시 기업 {len(visible):,}개"
    st.caption(f"{result_caption} · 좌표 변환 성공 {len(map_df):,}개")

    import pydeck as pdk

    if visible.empty:
        st.info("선택한 지역에 표시할 기업이 없습니다.")
        return

    visible["tooltip_text"] = visible.apply(
        lambda row: f"{row['entity_type']} · {row['name']}\\n{row['display_address']}",
        axis=1,
    )
    # 확대 10 수준에서 보이는 작은 역삼각형 크기로 고정합니다.
    triangle_delta = 0.00025
    visible["triangle"] = visible.apply(
        lambda row: [
            [float(row["longitude"]), float(row["latitude"]) - triangle_delta],
            [float(row["longitude"]) - triangle_delta, float(row["latitude"]) + triangle_delta * 0.65],
            [float(row["longitude"]) + triangle_delta, float(row["latitude"]) + triangle_delta * 0.65],
        ],
        axis=1,
    )
    center_lat = float(visible["latitude"].median())
    center_lon = float(visible["longitude"].median())
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=(11.0 if len(visible) == 1 else 9.5) if company_query else (6.2 if selected_region == "전체" else 8.5),
        ),
        tooltip={
            "html": (
                "<b>{name}</b><br/>"
                "유형: {entity_type}<br/>"
                "대표자: {representative}<br/>"
                "전화번호: {phone}<br/>"
                "주소: {display_address}"
            ),
            "style": {
                "backgroundColor": "#101010",
                "color": "white",
                "fontSize": "13px",
                "padding": "10px",
            },
        },
        layers=[
            pdk.Layer(
                "PolygonLayer",
                id="parent-company-points",
                data=visible,
                get_polygon="triangle",
                get_fill_color=[180, 100, 70, 190],
                get_line_color=[70, 50, 40, 220],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )
        ],
    )
    map_col, detail_col = st.columns([3.2, 1.2], gap="large")
    with map_col:
        selection = st.pydeck_chart(
            deck,
            height=620,
            selection_mode="single-object",
            on_select="rerun",
            key="parent_company_map",
        )

    selected_objects = []
    selected_indices = []
    if selection is not None:
        try:
            selection_state = selection.selection
            raw_objects = getattr(selection_state, "objects", []) or []
            raw_indices = getattr(selection_state, "indices", []) or []
        except AttributeError:
            selection_state = selection.get("selection", {})
            raw_objects = selection_state.get("objects", []) or []
            raw_indices = selection_state.get("indices", []) or []

        # Streamlit PyDeck state is grouped by layer ID:
        # {"parent-company-points": [{...}]} and {"parent-company-points": [3]}.
        if isinstance(raw_objects, dict):
            selected_objects = next(iter(raw_objects.values()), []) or []
        else:
            selected_objects = list(raw_objects)
        if isinstance(raw_indices, dict):
            selected_indices = next(iter(raw_indices.values()), []) or []
        else:
            selected_indices = list(raw_indices)

    if not selected_objects and not selected_indices:
        with detail_col:
            st.info("삼각형을 클릭하면 기업 정보가 표시됩니다.")
        return

    selected_object = selected_objects[0] if selected_objects else selected_indices[0]
    selected_id = ""
    if isinstance(selected_object, dict):
        selected_crno = str(selected_object.get("crno") or "")
        selected_id = str(selected_object.get("id") or selected_object.get("map_id") or "")
        selected_name = str(selected_object.get("name", ""))
    elif isinstance(selected_object, int) or (isinstance(selected_object, str) and selected_object.isdigit()):
        selected_index = int(selected_object)
        selected_row = visible.iloc[selected_index] if 0 <= selected_index < len(visible) else None
        selected_crno = str(selected_row.get("crno", "")) if selected_row is not None else ""
        selected_id = str(selected_row.get("id", "")) if selected_row is not None else ""
        selected_name = str(selected_row.get("name", "")) if selected_row is not None else ""
    else:
        parsed_object = None
        try:
            parsed_object = json.loads(str(selected_object))
        except (TypeError, json.JSONDecodeError):
            pass
        if isinstance(parsed_object, dict):
            selected_crno = str(parsed_object.get("crno", ""))
            selected_id = str(parsed_object.get("id") or parsed_object.get("map_id") or "")
            selected_name = str(parsed_object.get("name", ""))
        else:
            selected_crno = ""
            selected_name = str(selected_object)
    selected_rows = visible[visible["crno"].astype(str) == selected_crno]
    if selected_rows.empty and selected_id:
        selected_rows = visible[visible["id"].astype(str) == selected_id]
    if selected_rows.empty:
        selected_rows = visible[visible["name"] == selected_name]
    if selected_rows.empty:
        with detail_col:
            st.info("선택한 기업 정보를 찾지 못했습니다.")
        return

    row = selected_rows.iloc[0]
    with detail_col:
        st.markdown(
            "<div style='background:#101010;color:white;border-radius:12px;padding:16px 18px;margin-top:8px;'>"
            f"<div style='font-size:0.78rem;color:#cfc7bf;margin-bottom:8px;'>선택한 {html.escape(str(row.get('entity_type') or '기업'))}</div>"
            f"<div style='font-size:1.15rem;font-weight:700;margin-bottom:12px;'>{html.escape(str(row.get('name', '-')))}</div>"
            f"<div>대표자: {html.escape(str(row.get('representative') or '-'))}</div>"
            f"<div>전화번호: {html.escape(str(row.get('phone') or '-'))}</div>"
            f"<div style='margin-top:8px;color:#d8d2cc;font-size:0.86rem;'>주소: {html.escape(str(row.get('display_address') or '-'))}</div>"
            f"<div style='color:#d8d2cc;font-size:0.86rem;'>지역: {html.escape(str(row.get('region') or '-'))}</div>"
            f"<div style='color:#d8d2cc;font-size:0.86rem;'>업종: {html.escape(str(row.get('sicNm') or '-'))}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_rag_demo() -> None:
    """RAG 탭: chatbot.py 의 Graph RAG 에이전트 챗봇을 그대로 붙인다."""
    st.markdown("## 자연어로 묻고<br>그래프로 답합니다", unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Neo4j 지식그래프를 Text2Cypher 로 조회하고 뉴스 기사를 벡터 검색으로 찾아 답변하며, 사용한 관계·기사와 도구를 함께 보여줍니다.</p>',
        unsafe_allow_html=True,
    )

    try:
        import chatbot

        # streamlit 은 임포트된 모듈을 항상 다시 불러오지는 않는다.
        # chatbot.py 를 고쳐도 서버 재시작 없이 반영되도록 매 실행마다 갱신한다.
        # (모듈 최상단은 상수·정규식뿐이라 비용이 거의 없고, 에이전트 임포트는 함수 안에서 일어난다.)
        importlib.reload(chatbot)
        render_chat_panel = chatbot.render_chat_panel
    except Exception as exc:  # 모듈 자체를 불러오지 못한 경우
        st.error(f"챗봇 모듈을 불러오지 못했습니다: {exc}")
        return

    # 다른 섹션에서 넘어온 질문은 한 번만 실행되도록 꺼내서 전달한다.
    pending = st.session_state.pop("rag_pending", None)

    try:
        render_chat_panel(
            state_key="rag_messages",
            placeholder="예) 신한금융지주의 종속기업을 알려줘",
            pending_prompt=pending,
        )
    except ImportError as exc:
        # 에이전트/드라이버 미설치, .env 누락 등으로 에이전트를 만들지 못한 경우
        st.error(f"Graph RAG 에이전트를 초기화하지 못했습니다: {exc}")
        st.caption("`.env` 의 OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 설정을 확인해 주세요.")


def render_architecture(content: dict[str, object]) -> None:
    st.markdown("## 데이터에서 그래프,<br>그리고 답변까지", unsafe_allow_html=True)
    markup = "".join(
        f"""
        <div class="row">
            <div class="idx">{item['step']}</div>
            <div class="title">{item['title']}</div>
            <div class="body">{item['body']}</div>
            <div class="tail"></div>
        </div>
        """
        for item in build_architecture_steps()
    )
    st.markdown(markup, unsafe_allow_html=True)
    st.markdown('<div class="rule soft"></div>', unsafe_allow_html=True)
    st.markdown('<span class="kicker">stack</span>', unsafe_allow_html=True)
    pills = "".join(f"<span>{item}</span>" for item in content["stack"])
    st.markdown(f'<div class="stack">{pills}</div>', unsafe_allow_html=True)


def render_sidebar(content: dict[str, object]) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sb-mark">{content['wordmark']}</div>
            <span class="sb-label">Service</span>
            <div class="sb-value">{content['service_name']}</div>
            <span class="sb-label">Dataset</span>
            <div class="sb-value">금융위원회 기업기본정보</div>
            <span class="sb-label">Source</span>
            <div class="sb-value">공공데이터포털 OpenAPI</div>
            <span class="sb-label">Priority</span>
            <div class="sb-value">검색 → 상세 → 관계 그래프 → 관계 분류 → Graph RAG</div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="기업 연결 관계", page_icon="◍", layout="wide")
    content = get_site_content()
    render_sidebar(content)
    render_topbar(content)

    # st.tabs 는 코드에서 탭을 바꿀 수 없어, 질문 클릭 → RAG 이동을 위해 라디오로 내비게이션을 구성한다.
    section = st.radio(
        "섹션",
        SECTIONS,
        key="nav",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "Overview":
        render_problem_and_questions(content)
        st.markdown('<div class="rule soft"></div>', unsafe_allow_html=True)
        render_hero_band(content)
    elif section == "Graph":
        render_graph_explorer()
    elif section == "기업 위치 지도":
        render_company_map()
    elif section == "RAG":
        render_rag_demo()
    else:
        render_architecture(content)


if __name__ == "__main__":
    main()
