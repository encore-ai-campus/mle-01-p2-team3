import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

#도구 가져오기
from tools.text2cypher import select_names_in_graph,search_graph
#+ 담기
text2cypher_tools = [select_names_in_graph,search_graph]

# ====== 환경변수 불러오기 ======

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 .env에 설정되어 있지 않습니다."
    )


# ====== 기본 에이전트 생성 ======

# 모델 설정
luna_model = ChatOpenAI(
    model="gpt-5.6-luna",
    use_responses_api=True,
    output_version="responses/v1",
)

#전체 에이전트 프롬프트
SYSTEM_PROMPT = """
사용자의 기업 관련 질문을 분석하고
필요한 도구를 선택하여 답변합니다.

도구의 역할을 확인하고 필요한 경우 여러 도구를 순서대로 사용하세요.

- select_names_in_graph:
  사용자가 언급한 기업명, 지역명, 업종명이
  그래프에 실제 존재하는지 확인합니다.

- search_graph:
  Neo4j 그래프에서 관계와 속성을 조회합니다.
  조회 전용 Cypher를 작성하여 사용합니다.
"""

company_data_agent = create_agent(
    model = luna_model,
    tools = text2cypher_tools,
    system_prompt = SYSTEM_PROMPT
)


# ====== 여기는 테스트 구문입니다. ======

#테스트 해보기
human_prompt= input("기업 정보 도우미. 무엇이든 물어보세요: ")

#진행하기
result = company_data_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": human_prompt,
            }
        ]
})

# 사용한 도구 확인
print("\n[사용한 도구]")

used_tools = []

for message in result["messages"]:
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            used_tools.append(tool_name)
            print(f"- {tool_name}")

if not used_tools:
    print("- 사용된 도구 없음")


# 최종 답변
print("\n[최종 답변]")
print(result["messages"][-1].content)