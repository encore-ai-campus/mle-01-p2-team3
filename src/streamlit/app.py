from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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
        "tagline": "기업 관계를 지식 그래프로 연결해 산업 생태계와 잠재 고객사를 빠르게 탐색하는 B2B 리서치 서비스",
        "summary": "영업/마케팅 담당자가 기업명, 산업, 키워드, 관계 유형을 한 화면에서 탐색하고 그래프 기반 질의응답으로 다음 접점을 발견하도록 돕습니다.",
        "core_questions": [
            {"question": "특정 기업과 연결된 계열사는 어디인가?", "feature": "지식그래프 탐색"},
            {"question": "이 기업과 유사한 일을 하는 기업은 어디인가?", "feature": "유사 기업 추천"},
            {"question": "특정 산업에서 영업 대상이 될 만한 기업은 어디인가?", "feature": "산업별 기업 탐색 및 필터링"},
            {"question": "해당 기업의 규모와 최근 상황은 어떤가?", "feature": "그래프 RAG"},
        ],
        "stack": ["pandas", "OpenAI Embeddings", "OpenAI LLM", "Neo4j", "Cypher", "Streamlit"],
    }


def get_feature_rows() -> list[dict[str, str]]:
    return [
        {"Priority": "P0", "Feature": "기업 검색", "Description": "기업명, 산업, 키워드로 기업을 검색하고 기본 정보를 확인합니다.", "Owner": "이건호"},
        {"Priority": "P0", "Feature": "기업 상세 페이지", "Description": "산업, 주요 키워드, 관련 기업, 관계 유형을 한 화면에 제공합니다.", "Owner": "김수민"},
        {"Priority": "P0", "Feature": "기업 관계 그래프", "Description": "기업 간 연결 구조를 노드와 엣지로 시각화합니다.", "Owner": "엄가현"},
        {"Priority": "P0", "Feature": "관계 유형 분류", "Description": "계열사, 종속 기업, 공급망, 협력사 등 관계 유형을 구분합니다.", "Owner": "송진명"},
        {"Priority": "P0", "Feature": "기업 데이터 전처리", "Description": "중복, 누락, 짧은 문서를 정제해 그래프 품질을 높입니다.", "Owner": "김동석"},
        {"Priority": "P0", "Feature": "자연어 질의 응답", "Description": "질문을 Cypher/RAG 흐름으로 연결해 답변합니다.", "Owner": "이건호"},
        {"Priority": "P1", "Feature": "산업 생태계 맵", "Description": "산업별 핵심 기업과 주변 공급망 구조를 탐색합니다.", "Owner": "엄가현"},
        {"Priority": "P1", "Feature": "유사 기업 추천", "Description": "산업, 키워드, 관계 구조가 비슷한 기업을 추천합니다.", "Owner": "엄가현"},
        {"Priority": "P2", "Feature": "관계 근거 제공", "Description": "뉴스, 공시, 웹 문서에서 추출한 근거 문장을 함께 보여줍니다.", "Owner": "송진명"},
    ]


def build_sample_graph() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes = [
        CompanyNode("삼성전자", "핵심기업", 34, 420, 210, "반도체, 모바일, 디스플레이 생태계의 중심 기업"),
        CompanyNode("삼성SDI", "계열사", 23, 220, 120, "배터리 및 전자재료 제조"),
        CompanyNode("삼성전기", "계열사", 22, 230, 310, "전자부품 및 모듈 공급"),
        CompanyNode("현대자동차", "유사기업", 28, 640, 130, "모빌리티 제조 및 공급망 운영"),
        CompanyNode("LG에너지솔루션", "협력사", 24, 660, 320, "배터리 공급 및 에너지 솔루션"),
        CompanyNode("소재부품 협력사 A", "공급망", 18, 420, 420, "반도체 소재 및 부품 공급 후보"),
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
    return """
    <style>
    :root {
        --ink: #161616;
        --muted: #6d6a64;
        --paper: #fffaf0;
        --panel: #ffffff;
        --line: #1d1d1d;
        --yellow: #ffd84d;
        --coral: #ff6b57;
        --mint: #95dfc6;
        --blue: #7bb7ff;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #161616; }
    [data-testid="stSidebar"] * { color: #fffaf0 !important; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    .cue-nav { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--line); padding-bottom: 1rem; margin-bottom: 2.2rem; }
    .brand-mark { font-size: 1.05rem; font-weight: 900; text-transform: lowercase; }
    .nav-links { color: var(--muted); font-size: .9rem; word-spacing: 1.2rem; }
    .hero { border-bottom: 2px solid var(--line); padding: 1rem 0 2.4rem; position: relative; }
    .hero h1 { font-size: clamp(3.2rem, 8vw, 7.2rem); line-height: .92; margin: 0; max-width: 980px; }
    .hero p { max-width: 720px; font-size: 1.2rem; color: var(--muted); margin-top: 1.4rem; }
    .sticker { display: inline-block; border: 2px solid var(--line); border-radius: 999px; padding: .32rem .78rem; background: var(--yellow); font-weight: 800; transform: rotate(-2deg); }
    .section-kicker { color: var(--coral); font-weight: 900; text-transform: uppercase; font-size: .78rem; letter-spacing: .08em; }
    .metric-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .85rem; margin: 1.5rem 0 2.2rem; }
    .metric-card, .question-card, .feature-card, .stack-pill { border: 2px solid var(--line); background: var(--panel); box-shadow: 5px 5px 0 var(--line); }
    .metric-card { padding: 1rem; min-height: 104px; }
    .metric-card b { font-size: 1.8rem; display:block; }
    .metric-card span { color: var(--muted); font-size: .88rem; }
    .question-card { padding: 1rem; min-height: 132px; margin-bottom: .9rem; }
    .question-card b { color: var(--ink); }
    .question-card small { color: var(--muted); }
    .feature-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
    .feature-card { padding: 1rem; min-height: 158px; }
    .priority { display:inline-block; background: var(--mint); border: 1.5px solid var(--line); padding: .12rem .45rem; font-size: .74rem; font-weight: 900; margin-bottom: .65rem; }
    .stack-row { display:flex; flex-wrap:wrap; gap:.7rem; margin-top:.8rem; }
    .stack-pill { padding:.55rem .8rem; background:#fff; font-weight:800; }
    .answer-box { border: 2px solid var(--line); background: #ffffff; padding: 1.1rem; box-shadow: 5px 5px 0 var(--blue); }
    .pipeline { display:grid; grid-template-columns: repeat(7, minmax(120px, 1fr)); gap:.8rem; overflow-x:auto; padding-bottom:.4rem; }
    .pipe-step { min-width: 120px; background:#fff; border:2px solid var(--line); padding:.8rem; position:relative; }
    .pipe-step b { display:block; margin-bottom:.4rem; }
    .pipe-step small { color:var(--muted); }
    .source-note { color: var(--muted); font-size: .86rem; margin-top: .4rem; }
    @media (max-width: 850px) {
        .metric-strip, .feature-grid { grid-template-columns: 1fr; }
        .hero h1 { font-size: 3.1rem; }
        .nav-links { display:none; }
    }
    </style>
    """


def render_graph_svg(nodes: list[dict[str, object]], edges: list[dict[str, object]], selected_relation: str) -> str:
    node_lookup = {node["id"]: node for node in nodes}
    color_by_group = {
        "핵심기업": "#ff6b57",
        "계열사": "#ffd84d",
        "유사기업": "#7bb7ff",
        "협력사": "#95dfc6",
        "공급망": "#ffffff",
    }
    visible_edges = [edge for edge in edges if selected_relation == "전체" or edge["relation"] == selected_relation]
    edge_markup = []
    for edge in visible_edges:
        source = node_lookup[edge["source"]]
        target = node_lookup[edge["target"]]
        label_x = (source["x"] + target["x"]) / 2
        label_y = (source["y"] + target["y"]) / 2
        edge_markup.append(
            f'<line x1="{source["x"]}" y1="{source["y"]}" x2="{target["x"]}" y2="{target["y"]}" stroke="#161616" stroke-width="2" stroke-dasharray="7 5" />'
            f'<rect x="{label_x - 32}" y="{label_y - 13}" width="64" height="24" rx="12" fill="#fffaf0" stroke="#161616" stroke-width="1.5" />'
            f'<text x="{label_x}" y="{label_y + 5}" text-anchor="middle" font-size="12" font-weight="800">{html.escape(str(edge["relation"]))}</text>'
        )
    node_markup = []
    for node in nodes:
        fill = color_by_group.get(str(node["group"]), "#ffffff")
        node_markup.append(
            f'<circle cx="{node["x"]}" cy="{node["y"]}" r="{node["size"]}" fill="{fill}" stroke="#161616" stroke-width="3" />'
            f'<text x="{node["x"]}" y="{node["y"] + int(node["size"]) + 24}" text-anchor="middle" font-size="14" font-weight="900">{html.escape(str(node["id"]))}</text>'
            f'<text x="{node["x"]}" y="{node["y"] + int(node["size"]) + 42}" text-anchor="middle" font-size="11" fill="#6d6a64">{html.escape(str(node["group"]))}</text>'
        )
    return f"""
    <div style="border:2px solid #161616;background:#fff;box-shadow:6px 6px 0 #161616;overflow:auto;">
      <svg viewBox="0 0 860 520" width="100%" height="520" role="img" aria-label="기업 관계 그래프">
        <rect x="0" y="0" width="860" height="520" fill="#fffaf0" />
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """


def render_header(content: dict[str, object]) -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="cue-nav">
            <div class="brand-mark">cuestudio inspired / team3</div>
            <div class="nav-links">search graph rag architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="hero">
            <span class="sticker">B2B 산업 생태계 탐색</span>
            <h1>{content['service_name']}</h1>
            <p>{content['tagline']}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    st.markdown(
        """
        <div class="metric-strip">
            <div class="metric-card"><b>6</b><span>MVP 핵심 기능</span></div>
            <div class="metric-card"><b>4</b><span>영업/마케팅 핵심 질문</span></div>
            <div class="metric-card"><b>7</b><span>데이터-그래프-RAG 파이프라인</span></div>
            <div class="metric-card"><b>P0</b><span>검색, 상세, 그래프, 질의응답 우선</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_problem_and_questions(content: dict[str, object]) -> None:
    st.markdown('<span class="section-kicker">why it matters</span>', unsafe_allow_html=True)
    st.header("흩어진 기업 정보를 연결 가능한 영업 지도로 바꿉니다")
    left, right = st.columns([1.08, 1])
    with left:
        st.write(content["summary"])
        st.info("현재 방식: 포털 검색, 기업 홈페이지, 뉴스, 공시, 엑셀 리스트를 사람이 직접 비교하고 관계를 추론합니다.")
        st.warning("해결 방향: 기업, 산업, 제품, 거래/협력 관계를 그래프로 연결하고 자연어 질문을 그래프 질의로 전환합니다.")
    with right:
        for item in content["core_questions"]:
            st.markdown(
                f"""
                <div class="question-card">
                    <b>{item['question']}</b><br>
                    <small>답하는 기능: {item['feature']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feature_cards() -> None:
    st.markdown('<span class="section-kicker">product scope</span>', unsafe_allow_html=True)
    st.header("MVP 기능 구성")
    rows = get_feature_rows()
    cards = []
    for row in rows[:6]:
        cards.append(
            f"""
            <div class="feature-card">
                <span class="priority">{row['Priority']}</span>
                <h3>{row['Feature']}</h3>
                <p>{row['Description']}</p>
                <small>담당: {row['Owner']}</small>
            </div>
            """
        )
    st.markdown(f'<div class="feature-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    with st.expander("P1/P2 확장 기능 보기"):
        st.dataframe(pd.DataFrame(rows[6:]), hide_index=True, use_container_width=True)


def render_graph_explorer() -> None:
    st.markdown('<span class="section-kicker">knowledge graph</span>', unsafe_allow_html=True)
    st.header("기업 관계 그래프 탐색")
    nodes, edges = build_sample_graph()
    relation = st.segmented_control("관계 유형", ["전체", "계열사", "공급망", "유사기업", "협력사"], default="전체")
    components.html(render_graph_svg(nodes, edges, relation), height=560, scrolling=False)

    edge_df = pd.DataFrame(edges)
    if relation != "전체":
        edge_df = edge_df[edge_df["relation"] == relation]
    st.dataframe(
        edge_df.rename(columns={"source": "출발 기업", "target": "연결 기업", "relation": "관계 유형", "confidence": "신뢰도"}),
        hide_index=True,
        use_container_width=True,
    )


def render_rag_demo() -> None:
    st.markdown('<span class="section-kicker">graph rag</span>', unsafe_allow_html=True)
    st.header("자연어 질의 응답 프로토타입")
    question = st.text_input("질문", value="삼성전자와 연결된 계열사와 유사 기업은 어디인가요?")
    selected_company = st.selectbox("기준 기업", ["삼성전자", "삼성SDI", "현대자동차", "LG에너지솔루션"])
    if st.button("그래프 기반 답변 생성", type="primary"):
        st.markdown(
            f"""
            <div class="answer-box">
                <b>질문</b><br>{html.escape(question)}<br><br>
                <b>답변 초안</b><br>
                {html.escape(selected_company)} 기준으로 그래프를 조회하면 계열 관계, 공급망 관계, 유사기업 관계를 분리해 볼 수 있습니다.
                예시 데이터에서는 삼성전자와 삼성SDI/삼성전기가 계열사로 연결되고, LG에너지솔루션은 배터리 산업 키워드 기준의 유사기업 후보로 해석됩니다.<br><br>
                <b>예상 Cypher</b><br>
                <code>MATCH (c:Company {{name: '{html.escape(selected_company)}'}})-[r]-(n:Company) RETURN c, type(r), n, r.confidence</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("실서비스에서는 Text2Cypher 결과, Neo4j 조회 결과, 문서 근거 문장을 함께 결합합니다.")


def render_architecture(content: dict[str, object]) -> None:
    st.markdown('<span class="section-kicker">architecture</span>', unsafe_allow_html=True)
    st.header("아키텍처 구조도")
    steps = build_architecture_steps()
    step_markup = []
    for item in steps:
        step_markup.append(
            f"""
            <div class="pipe-step">
                <span class="priority">{item['step']}</span>
                <b>{item['title']}</b>
                <small>{item['body']}</small>
            </div>
            """
        )
    st.markdown(f'<div class="pipeline">{"".join(step_markup)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="source-note">흐름: 데이터 수집 -> 전처리 -> LLM 트리플 추출 -> Neo4j 통합 지식그래프 -> 품질 평가 -> Text2Cypher -> Streamlit</div>', unsafe_allow_html=True)
    st.subheader("기술 스택")
    st.markdown(f'<div class="stack-row">{"".join(f"<span class=\"stack-pill\">{item}</span>" for item in content["stack"])}</div>', unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Team 3")
        st.caption("B2B Knowledge Graph Explorer")
        st.divider()
        st.markdown("**데이터셋**")
        st.write("금융위원회_기업기본정보")
        st.markdown("**출처**")
        st.write("공공데이터포털 OpenAPI")
        st.markdown("**MVP 우선순위**")
        st.write("검색 -> 상세 -> 관계 그래프 -> 관계 분류 -> Graph RAG")


def main() -> None:
    st.set_page_config(page_title="B2B Knowledge Graph Explorer", page_icon="KG", layout="wide")
    content = get_site_content()
    render_sidebar()
    render_header(content)
    render_metrics()

    overview_tab, graph_tab, rag_tab, architecture_tab = st.tabs(["Overview", "Graph", "RAG", "Architecture"])
    with overview_tab:
        render_problem_and_questions(content)
        st.divider()
        render_feature_cards()
    with graph_tab:
        render_graph_explorer()
    with rag_tab:
        render_rag_demo()
    with architecture_tab:
        render_architecture(content)


if __name__ == "__main__":
    main()
