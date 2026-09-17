from __future__ import annotations

import base64
import html
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "clean"
FONT_WEIGHTS = {400: "Eulyoo1945-Regular", 600: "Eulyoo1945-SemiBold"}


def static_serving_enabled() -> bool:
    """정적 서빙이 켜져 있고 폰트 파일이 실제로 있는지 확인합니다."""
    try:
        enabled = bool(st.get_option("server.enableStaticServing"))
    except Exception:
        enabled = False
    return enabled and all((STATIC_DIR / f"{stem}.woff2").is_file() for stem in FONT_WEIGHTS.values())


@st.cache_data(show_spinner=False)
def font_data_uri(stem: str) -> str:
    path = STATIC_DIR / f"{stem}.woff2"
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/woff2;base64,{encoded}"


def font_face_css(url_prefix: str, weights: tuple[int, ...] = (400, 600)) -> str:
    """정적 서빙이 가능하면 URL로, 아니면 base64로 을유1945를 심습니다."""
    use_url = static_serving_enabled()
    blocks = []
    for weight in weights:
        stem = FONT_WEIGHTS[weight]
        if use_url:
            source = f"url('{url_prefix}{stem}.woff2') format('woff2'), url('{url_prefix}{stem}.woff') format('woff')"
        else:
            data_uri = font_data_uri(stem)
            if not data_uri:
                continue
            source = f"url({data_uri}) format('woff2')"
        blocks.append(
            "@font-face {"
            "font-family: 'Eulyoo1945';"
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
        overview = _read_clean("기업개요_최종.csv")
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("종속기업_정리.csv")
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
        overview = _read_clean("기업개요_최종.csv")
        affiliates = _read_clean("계열회사_전처리.csv")
        subsidiaries = _read_clean("종속기업_정리.csv")
        merged = _read_clean("모기업_계열사_종속기업_통합.csv")
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


@dataclass(frozen=True)
class CompanyNode:
    id: str
    group: str
    size: int
    x: int
    y: int
    description: str


@dataclass(frozen=True)
class CompanyEdge:
    source: str
    target: str
    relation: str
    confidence: float


def get_site_content() -> dict[str, object]:
    return {
        "service_name": "B2B Knowledge Graph Explorer",
        "wordmark": "기업 연결 관계",
        "topbar": "공시 데이터부터 지식그래프까지, 기업 간의 연결 관계를 한 화면에서 탐색하세요",
        "topbar_chip": "지식그래프 살펴보기",
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


def build_sample_graph() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes = [
        CompanyNode("삼성전자", "핵심기업", 46, 480, 268, "반도체, 모바일, 디스플레이 생태계의 중심 기업"),
        CompanyNode("삼성SDI", "계열사", 30, 196, 132, "배터리 및 전자재료 제조"),
        CompanyNode("삼성전기", "계열사", 28, 182, 392, "전자부품 및 모듈 공급"),
        CompanyNode("현대자동차", "유사기업", 34, 776, 148, "모빌리티 제조 및 공급망 운영"),
        CompanyNode("LG에너지솔루션", "협력사", 31, 792, 386, "배터리 공급 및 에너지 솔루션"),
        CompanyNode("소재부품 협력사 A", "공급망", 24, 470, 476, "반도체 소재 및 부품 공급 후보"),
    ]
    edges = [
        CompanyEdge("삼성전자", "삼성SDI", "계열사", 0.94),
        CompanyEdge("삼성전자", "삼성전기", "계열사", 0.92),
        CompanyEdge("삼성전자", "소재부품 협력사 A", "공급망", 0.81),
        CompanyEdge("현대자동차", "LG에너지솔루션", "협력사", 0.78),
        CompanyEdge("삼성SDI", "LG에너지솔루션", "유사기업", 0.86),
        CompanyEdge("삼성전자", "현대자동차", "유사기업", 0.67),
    ]
    return [node.__dict__ for node in nodes], [edge.__dict__ for edge in edges]


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
        --display: 'Eulyoo1945', 'Apple SD Gothic Neo', serif;
        --sans: 'Eulyoo1945', 'Inter', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
        --label: 'Inter', 'Eulyoo1945', sans-serif;
    }}

    .stApp {{ background: var(--paper); color: var(--ink); }}
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div {{ font-family: var(--sans); }}
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

    /* 상단 안내 바 */
    .topbar {{
        display:flex; align-items:center; justify-content:center; gap: 1rem; flex-wrap: wrap;
        background: var(--accent); color: var(--ink);
        margin: -1.6rem -100vw 0; padding: .72rem 100vw;
        font-size: .88rem;
    }}
    .topbar .chip {{
        border: 1px solid var(--ink); border-radius: 999px; padding: .18rem .8rem;
        font-family: var(--label); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
    }}

    /* 브랜드 + 탭 내비게이션 */
    .brandbar {{ padding: 1.6rem 0 0; }}
    .brandbar .mark {{ font-family: var(--display); font-size: 1.7rem; line-height: 1; letter-spacing: -0.02em; }}
    .stTabs {{ margin-top: -2.9rem; }}
    .stTabs [role="tablist"] {{ justify-content: flex-end; padding-bottom: .2rem; }}
    .rule {{ height:1px; background: var(--line-strong); margin: 0 0 3.4rem; }}
    .rule.soft {{ background: var(--line); margin: 2.6rem 0; }}

    /* 홈 3단 패널 */
    .panel {{ padding: 2.2rem 1.8rem 2rem 0; border-right: 1px solid var(--line); min-height: 320px; }}
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

    /* 우측 쇼케이스 */
    .showcase {{
        background: var(--accent); color: var(--ink);
        padding: 2.4rem 2.2rem; min-height: 320px;
        display:flex; flex-direction: column; justify-content: center;
    }}
    .showcase-kicker {{ font-family: var(--label); font-size: .68rem; letter-spacing: .24em; opacity: .75; }}
    .showcase-title {{
        font-size: clamp(1.9rem, 2.9vw, 3rem) !important; line-height: 1.2 !important;
        margin: 1.1rem 0 1.2rem !important; word-break: keep-all;
    }}
    .showcase-body {{ font-size: .95rem; line-height: 1.8; margin: 0; max-width: 34rem; }}
    .showcase-tags {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top: 1.8rem; }}
    .showcase-tags span {{
        border: 1px solid var(--ink); border-radius: 999px; padding: .28rem .85rem; font-size: .78rem;
    }}

    /* editorial rows */
    .kicker {{ font-family: var(--label); font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--mute); display:block; margin-bottom: 1.1rem; }}
    .row {{ display:grid; grid-template-columns: 78px 1.05fr 1.6fr 96px; gap: 1.6rem; align-items: baseline; padding: 1.35rem 0; border-top: 1px solid var(--line); }}
    .row:last-child {{ border-bottom: 1px solid var(--line); }}
    .row .idx {{ font-family: var(--label); font-size: .74rem; letter-spacing: .12em; color: var(--mute); }}
    .row .title {{ font-family: var(--display); font-size: 1.5rem; letter-spacing: -0.01em; }}
    .row .body {{ color: var(--mute); font-size: .93rem; line-height: 1.8; font-weight: 400; }}
    .row .tail {{ font-family: var(--label); text-align:right; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--mute); }}
    .row:hover .title {{ font-weight: 600; }}

    /* question list */
    .q {{ padding: 1.25rem 0; border-top: 1px solid var(--line); }}
    .q:last-child {{ border-bottom: 1px solid var(--line); }}
    .q b {{ font-weight: 600; font-size: 1.02rem; }}
    .q small {{ font-family: var(--label); display:block; margin-top: .45rem; color: var(--mute); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }}

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
        gap: 2.4rem !important; background: transparent !important; border-bottom: 1px solid var(--line) !important;
    }}
    .stTabs [role="tab"], button[data-baseweb="tab"] {{
        background: transparent !important; padding: 0 0 .85rem 0 !important;
    }}
    .stTabs [role="tab"] p, button[data-baseweb="tab"] p {{
        font-family: var(--label) !important; font-size: .76rem !important; letter-spacing: .18em !important;
        text-transform: uppercase; color: var(--mute) !important;
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
    [data-testid="stSidebar"] .sb-label {{ font-family: var(--label); font-size: .68rem; letter-spacing: .2em; text-transform: uppercase; color: var(--mute); display:block; margin-top: 1.6rem; }}
    [data-testid="stSidebar"] .sb-value {{ font-size: .95rem; margin-top: .3rem; line-height: 1.6; }}

    [data-testid="stDataFrame"] {{ border: 1px solid var(--line); }}

    @media (max-width: 1100px) {{
        /* 좁은 화면에서는 브랜드 아래로 탭이 내려오도록 되돌립니다. */
        .stTabs {{ margin-top: .6rem; }}
        .stTabs [role="tablist"] {{ justify-content: flex-start; }}
        .panel {{ border-right: 0; min-height: 0; padding: 1.6rem 0; border-bottom: 1px solid var(--line); }}
    }}

    @media (max-width: 900px) {{
        .row {{ grid-template-columns: 1fr; gap: .5rem; }}
        .row .tail {{ text-align:left; }}
    }}
    </style>
    """


def render_graph_svg(nodes: list[dict[str, object]], edges: list[dict[str, object]], selected_relation: str) -> str:
    lookup = {node["id"]: node for node in nodes}
    visible = [edge for edge in edges if selected_relation == "전체" or edge["relation"] == selected_relation]

    edge_markup: list[str] = []
    for edge in visible:
        source, target = lookup[edge["source"]], lookup[edge["target"]]
        sx, sy, tx, ty = source["x"], source["y"], target["x"], target["y"]
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1
        cx, cy = mx - dy / length * 46, my + dx / length * 46
        px, py = 0.25 * sx + 0.5 * cx + 0.25 * tx, 0.25 * sy + 0.5 * cy + 0.25 * ty
        label = html.escape(str(edge["relation"]))
        width = 16 + len(label) * 12
        edge_markup.append(
            f'<path d="M {sx} {sy} Q {cx:.1f} {cy:.1f} {tx} {ty}" fill="none" stroke="#cfccc4" stroke-width="1" />'
            f'<rect x="{px - width / 2:.1f}" y="{py - 11:.1f}" width="{width}" height="22" fill="#fbfaf8" />'
            f'<text x="{px:.1f}" y="{py + 4:.1f}" text-anchor="middle" font-size="11" letter-spacing="1.4" '
            f'fill="#8c887f" font-family="Eulyoo1945, serif">{label}</text>'
        )

    node_markup: list[str] = []
    for node in nodes:
        is_core = node["group"] == "핵심기업"
        fill = "#b4aaa1" if is_core else "#dbd6d1"
        radius = int(node["size"])
        group_label = html.escape(str(node["group"]))
        node_markup.append(
            f'<circle cx="{node["x"]}" cy="{node["y"]}" r="{radius}" fill="{fill}" />'
            f'<text x="{node["x"]}" y="{node["y"] + radius + 26}" text-anchor="middle" font-size="15" '
            f'fill="#101010" font-family="Eulyoo1945, serif">{html.escape(str(node["id"]))}</text>'
            f'<text x="{node["x"]}" y="{node["y"] + radius + 44}" text-anchor="middle" font-size="10" letter-spacing="1.6" '
            f'fill="#8c887f" font-family="Eulyoo1945, serif">{group_label}</text>'
        )

    return f"""
    <style>
    {font_face_css("/app/static/", weights=(400,))}
    body {{ margin: 0; }}
    </style>
    <div style="background:#fbfaf8;border-top:1px solid #101010;border-bottom:1px solid #e2dfd8;">
      <svg viewBox="0 0 960 560" width="100%" height="560" role="img" aria-label="기업 관계 지식 그래프">
        <rect x="0" y="0" width="960" height="560" fill="#fbfaf8" />
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """



def compact(markup: str) -> str:
    """줄바꿈/들여쓰기를 제거합니다. 빈 줄이 있으면 streamlit 의 마크다운 파서가 HTML 블록을 끊습니다."""
    return "".join(line.strip() for line in markup.splitlines())


def render_topbar(content: dict[str, object]) -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        compact(f"""
        <div class="topbar">
            <span>{content['topbar']}</span>
            <span class="chip">{content['topbar_chip']}</span>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_brand(content: dict[str, object]) -> None:
    st.markdown(
        compact(f"""
        <div class="brandbar">
            <div class="mark">{content['wordmark']}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_hero_band(content: dict[str, object]) -> None:
    """혁신의숲 홈 구조: 데이터 규모 · 연결이 많은 기업 · 그래프 비주얼."""
    stats = load_dataset_stats()
    ranking = load_top_connected(5)

    left, middle, right = st.columns([1, 1.05, 1.75], gap="large")

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
            <div class="panel">
                <h3 class="panel-title">연결 관계가 가장 많은 기업</h3>
                <ol class="ranking">{items}</ol>
                <div class="panel-foot">계열사 + 종속기업 수 기준 · 기업명이 확인되는 모기업만 집계</div>
            </div>
            """
        else:
            body = '<div class="panel"><h3 class="panel-title">집계할 관계 데이터가 없습니다</h3></div>'
        st.markdown(compact(body), unsafe_allow_html=True)

    with right:
        st.markdown(
            compact(f"""
            <div class="showcase">
                <span class="showcase-kicker">KNOWLEDGE GRAPH</span>
                <h2 class="showcase-title">{content['hero_lines'][0]}</h2>
                <p class="showcase-body">{content['tagline']}</p>
                <div class="showcase-tags">
                    {''.join(f'<span>{item}</span>' for item in content['statement'])}
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


def render_problem_and_questions(content: dict[str, object]) -> None:
    st.markdown('<span class="kicker">why it matters</span>', unsafe_allow_html=True)
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
        cards = "".join(
            f'<div class="q"><b>{item["question"]}</b><small>{item["feature"]}</small></div>'
            for item in content["core_questions"]
        )
        st.markdown(cards, unsafe_allow_html=True)


def render_graph_explorer() -> None:
    st.markdown('<span class="kicker">knowledge graph</span>', unsafe_allow_html=True)
    st.markdown("## 기업 관계 그래프 탐색")
    nodes, edges = build_sample_graph()
    relation = st.segmented_control("관계 유형", ["전체", "계열사", "공급망", "유사기업", "협력사"], default="전체")
    relation = relation or "전체"
    st.markdown(
        """
        <div class="legend">
            <span><i style="background:#b4aaa1;border-color:#b4aaa1"></i>핵심기업</span>
            <span><i style="background:#dbd6d1;border-color:#dbd6d1"></i>계열사 · 유사기업 · 협력사 · 공급망</span>
            <span>Edge — 관계 유형 / Confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(render_graph_svg(nodes, edges, relation), height=580, scrolling=False)


def render_rag_demo() -> None:
    st.markdown('<span class="kicker">graph rag</span>', unsafe_allow_html=True)
    st.markdown("## 자연어로 묻고<br>그래프로 답합니다", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        question = st.text_input("질문", value="삼성전자와 연결된 계열사와 유사 기업은 어디인가요?")
    with right:
        selected_company = st.selectbox("기준 기업", ["삼성전자", "삼성SDI", "현대자동차", "LG에너지솔루션"])
    if st.button("답변 생성"):
        company = html.escape(selected_company)
        st.markdown(
            f"""
            <div class="answer">
                <span class="label">Question</span>
                <p>{html.escape(question)}</p>
                <span class="label">Draft answer</span>
                <p>{company} 기준으로 그래프를 조회하면 계열 관계, 공급망 관계, 유사기업 관계를 분리해 볼 수 있습니다.
                예시 데이터에서는 삼성전자와 삼성SDI · 삼성전기가 계열사로 연결되고,
                LG에너지솔루션은 배터리 산업 키워드 기준의 유사기업 후보로 해석됩니다.</p>
                <span class="label">Generated cypher</span>
                <code>MATCH (c:Company {{name: '{company}'}})-[r]-(n:Company)
RETURN c, type(r), n, r.confidence
ORDER BY r.confidence DESC</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("실서비스에서는 Text2Cypher 결과, Neo4j 조회 결과, 문서 근거 문장을 함께 결합합니다.")


def render_architecture(content: dict[str, object]) -> None:
    st.markdown('<span class="kicker">architecture</span>', unsafe_allow_html=True)
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
            <div style="font-family:'Instrument Serif',serif;font-size:2rem;line-height:1;">{content['wordmark']}</div>
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
    render_brand(content)

    overview_tab, graph_tab, rag_tab, architecture_tab = st.tabs(["Overview", "Graph", "RAG", "Architecture"])
    with overview_tab:
        render_hero_band(content)
        st.markdown('<div class="rule soft"></div>', unsafe_allow_html=True)
        render_problem_and_questions(content)
    with graph_tab:
        render_graph_explorer()
    with rag_tab:
        render_rag_demo()
    with architecture_tab:
        render_architecture(content)


if __name__ == "__main__":
    main()
