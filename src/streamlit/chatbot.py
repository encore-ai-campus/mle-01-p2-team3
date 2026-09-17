from __future__ import annotations

import ast
import html
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MAX_GRAPH_EVIDENCE_ROWS = 20


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_answer(result: dict[str, Any]) -> str:
    """Agent 결과에서 사용자에게 보여줄 최종 텍스트만 추출한다."""
    messages = result.get("messages", [])
    if not messages:
        return ""

    content = _get_value(messages[-1], "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_blocks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                text_blocks.append(str(block["text"]))
        return "\n".join(text_blocks).strip()

    return str(content).strip() if content is not None else ""


def extract_used_tools(result: dict[str, Any]) -> list[str]:
    """Agent 실행 중 호출된 Tool 이름을 중복 없이 호출 순서대로 반환한다."""
    used_tools: list[str] = []
    seen: set[str] = set()

    for message in result.get("messages", []):
        tool_calls = _get_value(message, "tool_calls", None)
        if not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_name = _get_value(tool_call, "name", None)
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            used_tools.append(tool_name)

    return used_tools


def build_agent_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Streamlit 표시용 필드를 제거하고 Agent에 전달할 대화 기록을 만든다."""
    agent_messages: list[dict[str, str]] = []

    for message in messages:
        if message.get("is_error"):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue

        agent_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return agent_messages


def _parse_tool_content(content: Any) -> dict[str, Any]:
    """ToolMessage content를 가능한 범위에서 dict로 변환한다."""
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if text:
                parsed = _parse_tool_content(text)
                if parsed:
                    return parsed
        return {}

    if not isinstance(content, str):
        return {}

    text = content.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def extract_graph_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    """search_graph 결과 중 source-relationship-target 관계 행만 추출한다."""
    graph_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for message in result.get("messages", []):
        if _get_value(message, "name", None) != "search_graph":
            continue

        payload = _parse_tool_content(
            _get_value(message, "content", None)
        )
        results = payload.get("results", [])

        if not isinstance(results, list):
            continue

        for row in results:
            if not isinstance(row, dict):
                continue

            source = str(row.get("source", "")).strip()
            relationship = str(row.get("relationship", "")).strip()
            target = str(row.get("target", "")).strip()

            if not source or not relationship or not target:
                continue

            source_case = str(row.get("source_case", ""))
            source_row = str(row.get("source_row", ""))
            dedup_key = (source, relationship, target, f"{source_case}:{source_row}")

            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            graph_rows.append(
                {
                    "source": source,
                    "source_type": row.get("source_type"),
                    "relationship": relationship,
                    "target": target,
                    "target_type": row.get("target_type"),
                    "evidence": row.get("evidence"),
                    "source_case": row.get("source_case"),
                    "source_row": row.get("source_row"),
                }
            )

    return graph_rows


def select_display_graph_evidence(
    answer: str,
    graph_rows: list[dict[str, Any]],
    max_rows: int = MAX_GRAPH_EVIDENCE_ROWS,
) -> list[dict[str, Any]]:
    """답변에 실제 언급된 target을 우선하고, 화면 표시량을 제한한다."""
    if max_rows <= 0:
        return []

    matched_rows: list[dict[str, Any]] = []

    for row in graph_rows:
        target = str(row.get("target", "")).strip()
        if target and target in answer:
            matched_rows.append(row)

    if matched_rows:
        return matched_rows[:max_rows]

    return graph_rows[:max_rows]


def build_graph_visualization_data(
    graph_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Graph DB 관계 행을 PyVis용 노드/엣지 데이터로 변환한다."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for row in graph_rows:
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        relationship = str(row.get("relationship", "")).strip()

        if not source or not target or not relationship:
            continue

        source_type = str(row.get("source_type") or "Unknown")
        target_type = str(row.get("target_type") or "Unknown")

        source_id = f"{source_type}::{source}"
        target_id = f"{target_type}::{target}"

        nodes_by_id.setdefault(
            source_id,
            {
                "id": source_id,
                "label": source,
                "group": source_type,
                "title": (
                    f"<b>{html.escape(source)}</b><br>"
                    f"타입: {html.escape(source_type)}"
                ),
            },
        )
        nodes_by_id.setdefault(
            target_id,
            {
                "id": target_id,
                "label": target,
                "group": target_type,
                "title": (
                    f"<b>{html.escape(target)}</b><br>"
                    f"타입: {html.escape(target_type)}"
                ),
            },
        )

        edge_title = [f"관계: {html.escape(relationship)}"]

        evidence = row.get("evidence")
        if evidence not in (None, ""):
            edge_title.append(f"근거: {html.escape(str(evidence))}")

        source_case = row.get("source_case")
        if source_case not in (None, ""):
            edge_title.append(f"source_case: {html.escape(str(source_case))}")

        source_row = row.get("source_row")
        if source_row not in (None, ""):
            edge_title.append(f"source_row: {html.escape(str(source_row))}")

        edges.append(
            {
                "from": source_id,
                "to": target_id,
                "label": relationship,
                "title": "<br>".join(edge_title),
                "arrows": "to",
            }
        )

    return list(nodes_by_id.values()), edges


def populate_pyvis_network(
    network: Any,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    """PyVis API 형식에 맞춰 노드와 엣지를 Network에 추가한다."""
    for node in nodes:
        node_options = {
            key: value
            for key, value in node.items()
            if key != "id"
        }
        network.add_node(
            node["id"],
            **node_options,
        )

    for edge in edges:
        edge_options = {
            key: value
            for key, value in edge.items()
            if key not in {"from", "to"}
        }
        network.add_edge(
            edge["from"],
            edge["to"],
            **edge_options,
        )


def render_tool_expander(tool_names: list[str]) -> None:
    import streamlit as st

    if not tool_names:
        return

    with st.expander("사용한 도구"):
        for tool_name in tool_names:
            st.markdown(f"- `{tool_name}`")


def render_graph_network(
    graph_rows: list[dict[str, Any]],
    total_rows: int | None = None,
) -> None:
    """답변에 사용된 Graph DB 관계를 인터랙티브 네트워크로 표시한다."""
    import streamlit as st

    if not graph_rows:
        return

    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
    except ImportError:
        st.warning(
            "Graph DB 시각화를 사용하려면 `pyvis`가 필요합니다. "
            "프로젝트 루트에서 `uv add pyvis`를 실행해 주세요."
        )
        return

    nodes, edges = build_graph_visualization_data(graph_rows)
    if not nodes or not edges:
        return

    total = total_rows if total_rows is not None else len(graph_rows)
    shown = len(graph_rows)

    st.markdown("#### 관련 Graph DB")

    if total > shown:
        st.caption(
            f"조회된 관계 {total}개 중 답변과 관련된 {shown}개 관계만 표시합니다."
        )
    else:
        st.caption(f"이번 답변에 사용된 관계 {shown}개를 표시합니다.")

    network = Network(
        height="420px",
        width="100%",
        directed=True,
        cdn_resources="in_line",
    )

    populate_pyvis_network(
        network,
        nodes,
        edges,
    )

    network.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 250
            },
            "barnesHut": {
              "gravitationalConstant": -12000,
              "springLength": 170,
              "springConstant": 0.04
            }
          },
          "nodes": {
            "shape": "dot",
            "size": 22,
            "font": {
              "size": 15
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "font": {
              "size": 11,
              "align": "middle"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            }
          }
        }
        """
    )

    graph_html = network.generate_html(notebook=False)
    components.html(
        graph_html,
        height=450,
        scrolling=False,
    )


def render_graph_evidence(
    graph_rows: list[dict[str, Any]],
    total_rows: int | None = None,
) -> None:
    import streamlit as st

    if not graph_rows:
        return

    total = total_rows if total_rows is not None else len(graph_rows)
    shown = len(graph_rows)

    title = f"상세 근거 ({shown}개 표시"
    if total > shown:
        title += f" / 전체 {total}개"
    title += ")"

    with st.expander(title):
        table_rows = []

        for row in graph_rows:
            table_rows.append(
                {
                    "시작 노드": row.get("source", ""),
                    "관계": row.get("relationship", ""),
                    "연결 노드": row.get("target", ""),
                    "근거": row.get("evidence", ""),
                    "원본 행": row.get("source_row", ""),
                }
            )

        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
        )

        if total > shown:
            st.caption(
                f"조회된 관계가 많아 {total}개 중 {shown}개만 표시합니다. "
                "답변에 직접 언급된 연결 노드를 우선해서 보여줍니다."
            )


_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODE_SPAN_PATTERN = re.compile(r"(`+[^`]*`+)")


def render_answer_markdown(text: str) -> None:
    """답변을 렌더링한다.

    마크다운(CommonMark)은 **강조** 뒤에 곧바로 글자가 오면 닫는 기호로 보지 않는다.
    한국어는 '**휴양콘도 운영업**을' 처럼 조사가 붙어서 별표가 그대로 노출되므로,
    코드 스팬 밖의 ** 쌍만 <strong> 으로 바꿔서 렌더링한다.
    """
    import streamlit as st

    parts = []
    for segment in _CODE_SPAN_PATTERN.split(str(text)):
        if segment.startswith("`"):
            parts.append(segment)  # 코드 스팬은 그대로 둔다
            continue
        escaped = html.escape(segment, quote=False)
        parts.append(_BOLD_PATTERN.sub(r"<strong>\g<1></strong>", escaped))

    st.markdown("".join(parts), unsafe_allow_html=True)


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    import streamlit as st

    for message in messages:
        role = message["role"]

        with st.chat_message(role):
            if message.get("is_error"):
                st.error(message["content"])
            else:
                render_answer_markdown(message["content"])

            if role == "assistant":
                graph_rows = message.get("graph_evidence", [])
                graph_total = message.get("graph_evidence_total")

                render_graph_network(
                    graph_rows,
                    graph_total,
                )
                render_graph_evidence(
                    graph_rows,
                    graph_total,
                )
                render_tool_expander(
                    message.get("used_tools", [])
                )


def render_chat_panel(
    state_key: str = "messages",
    placeholder: str = "기업 정보를 질문해 주세요.",
    reset_label: str | None = "대화 초기화",
    pending_prompt: str | None = None,
) -> None:
    """페이지 설정 없이 챗봇 UI 만 그린다. app.py 의 탭 안에서도 재사용한다."""
    import streamlit as st
    from src.agent.agent import company_data_agent

    if state_key not in st.session_state:
        st.session_state[state_key] = []

    if reset_label and st.button(reset_label, key=f"{state_key}_reset"):
        st.session_state[state_key] = []
        st.rerun()

    render_chat_history(st.session_state[state_key])

    typed_prompt = st.chat_input(placeholder, key=f"{state_key}_input")
    # 다른 화면에서 넘겨준 질문(pending_prompt)도 입력과 동일하게 처리한다.
    human_prompt = typed_prompt or pending_prompt

    if not human_prompt:
        return

    st.session_state[state_key].append(
        {
            "role": "user",
            "content": human_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(human_prompt)

    with st.chat_message("assistant"):
        try:
            agent_messages = build_agent_messages(
                st.session_state[state_key]
            )

            with st.spinner("조회 중..."):
                result = company_data_agent.invoke(
                    {
                        "messages": agent_messages,
                    }
                )

            answer = extract_answer(result)
            used_tools = extract_used_tools(result)

            all_graph_evidence = extract_graph_evidence(result)
            display_graph_evidence = select_display_graph_evidence(
                answer,
                all_graph_evidence,
            )

            if not answer:
                answer = "답변을 생성하지 못했습니다. 다시 질문해 주세요."

            render_answer_markdown(answer)

            render_graph_network(
                display_graph_evidence,
                len(all_graph_evidence),
            )

            render_graph_evidence(
                display_graph_evidence,
                len(all_graph_evidence),
            )

            render_tool_expander(used_tools)

            st.session_state[state_key].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "used_tools": used_tools,
                    "graph_evidence": display_graph_evidence,
                    "graph_evidence_total": len(all_graph_evidence),
                }
            )

        except Exception as exc:
            error_message = f"조회 중 오류가 발생했습니다: {exc}"
            st.error(error_message)

            st.session_state[state_key].append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "used_tools": [],
                    "graph_evidence": [],
                    "graph_evidence_total": 0,
                    "is_error": True,
                }
            )


def main() -> None:
    """chatbot.py 를 단독 실행할 때 쓰는 진입점."""
    import streamlit as st

    st.set_page_config(
        page_title="기업 정보 챗봇",
        page_icon="💬",
    )
    st.title("기업 정보 챗봇")
    render_chat_panel()


if __name__ == "__main__":
    main()
