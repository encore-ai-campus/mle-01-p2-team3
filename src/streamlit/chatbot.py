from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.agent.agent import company_data_agent


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


def build_agent_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
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




def render_tool_expander(tool_names: list[str]) -> None:
    if not tool_names:
        return

    with st.expander("사용한 도구"):
        for tool_name in tool_names:
            st.markdown(f"- `{tool_name}`")


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        role = message["role"]

        with st.chat_message(role):
            if message.get("is_error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

            if role == "assistant":
                render_tool_expander(message.get("used_tools", []))


def main() -> None:
    st.set_page_config(
        page_title="기업 정보 챗봇",
        page_icon="💬",
    )
    st.title("기업 정보 챗봇")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    render_chat_history(st.session_state.messages)

    human_prompt = st.chat_input("기업 정보를 질문해 주세요.")
    if not human_prompt:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": human_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(human_prompt)

    with st.chat_message("assistant"):
        try:
            agent_messages = build_agent_messages(st.session_state.messages)

            with st.spinner("조회 중..."):
                result = company_data_agent.invoke(
                    {
                        "messages": agent_messages,
                    }
                )

            answer = extract_answer(result)
            used_tools = extract_used_tools(result)

            if not answer:
                answer = "답변을 생성하지 못했습니다. 다시 질문해 주세요."

            st.markdown(answer)
            render_tool_expander(used_tools)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "used_tools": used_tools,
                }
            )

        except Exception as exc:
            error_message = f"조회 중 오류가 발생했습니다: {exc}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "used_tools": [],
                    "is_error": True,
                }
            )


if __name__ == "__main__":
    main()
