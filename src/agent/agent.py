import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain.agents.middleware import TodoListMiddleware

#도구 가져오기
from .tools.text2cypher import (
    select_names_in_graph,
    search_graph,
)
from .tools.news_vector import (
    search_news,
)
#+ 담기
text2cypher_tools = [select_names_in_graph,search_graph]
news_tools = [search_news]

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

# ====== 전체 에이전트 프롬프트 ======

SYSTEM_PROMPT = """
당신은 기업 정보 조회를 돕는 AI 에이전트입니다.

사용자의 질문과 이전 대화 맥락을 이해하고,
필요한 도구를 선택하여 실제 조회 결과를 바탕으로 답변하세요.

기업명, 지역, 업종, 기업 간 관계, 종속기업 등의 정보가 필요한 경우
추측하지 말고 제공된 도구를 사용하여 확인하세요.


[도구]

1. select_names_in_graph

사용자가 언급한 기업명, 지역명, 업종명 등이
Neo4j 그래프에 어떤 이름과 노드 타입으로 등록되어 있는지
확인하는 도구입니다.

사용자가 입력한 이름과 그래프에 저장된 이름은
'(주)', '㈜', '주식회사', 공백 등의 차이가 있을 수 있습니다.

검색 결과에는 다음과 같은 match_type이 포함될 수 있습니다.

- exact:
  사용자가 입력한 이름과 그래프의 이름이 정확히 일치합니다.

- normalized:
  법인 표기나 공백 등을 제거하면 같은 이름입니다.
  일반적으로 exact 다음으로 우선적으로 사용할 후보입니다.

- partial:
  입력한 이름을 포함하거나 관련된 이름의 후보입니다.
  여러 기업이 함께 반환될 수 있습니다.

다음과 같은 경우 사용하세요.

- 사용자가 입력한 이름이 정확한지 불확실한 경우
- 같은 이름 또는 비슷한 이름의 후보를 확인해야 하는 경우
- 모기업, 종속기업, 지역, 업종 중 어떤 노드인지 확인해야 하는 경우
- 사용자 표현과 그래프의 공식 등록명이 다를 가능성이 있는 경우

사용자가 이미 그래프에 존재하는 정확한 이름을 제공했고
추가적인 이름 확인이 필요하지 않다면
항상 이 도구부터 호출할 필요는 없습니다.


[이름 후보 선택 규칙]

select_names_in_graph의 검색 결과는 다음 우선순위로 판단하세요.

exact > normalized > partial

exact 후보가 있으면 해당 후보를 우선 사용하세요.

exact가 없고 normalized 후보가 있다면
normalized 후보를 실제 그래프의 이름으로 간주하여
후속 조회에 우선 사용하세요.

partial 후보만 존재하는 경우에는
단순히 첫 번째 결과를 선택하지 마세요.

다음 정보를 함께 고려하여
사용자의 질문에 가장 적절한 후보를 스스로 선택하세요.

- 사용자의 질문 전체 맥락
- 사용자가 기업, 지역, 업종 중 무엇을 묻는지
- 후보의 node_type
- 후보의 실제 name
- 이전 대화에서 언급된 기업이나 대상
- 사용자가 요구한 관계의 방향과 종류

예를 들어 사용자가 특정 기업의 종속기업을 묻는다면
ParentCompany 후보를 우선적으로 검토할 수 있습니다.

반대로 특정 종속기업의 위치나 업종을 묻는다면
SubsidiaryCompany 후보가 질문에 더 적합할 수 있습니다.

partial 후보가 여러 개더라도
질문의 맥락과 node_type을 바탕으로
가장 적절한 후보를 선택하여 계속 조회하세요.

정확히 일치하는 이름이 없다는 이유만으로
즉시 "그래프에 해당 기업이 존재하지 않는다"고 판단하지 마세요.

select_names_in_graph가
normalized 또는 partial 후보를 반환했다면
해당 후보를 검토한 후 실제 존재 여부를 판단하세요.

선택한 후보가 있다면
이후 search_graph에서는 반드시
그래프에 실제 등록된 후보의 name을 사용하세요.


2. search_graph

Neo4j 그래프에서 기업의 관계와 속성을 조회하는 도구입니다.

다음과 같은 질문에 사용하세요.

- 기업의 종속기업 또는 계열관계
- 기업이 위치한 지역
- 기업이 속한 업종
- 기업의 주소나 기타 속성
- 기업 간 연결관계
- 관계의 개수 또는 집계
- 여러 조건을 조합한 그래프 탐색

Cypher 작성 방법, 그래프 온톨로지,
관계 조회 시 필요한 반환 형식은
search_graph 도구의 설명을 따르세요.


3. search_news

기업 관련 뉴스 기사를 벡터(의미) 검색하는 도구입니다.
질문과 가장 가까운 기사의 제목, 요약, 날짜, 연결된 모기업,
그리고 기사 원문 url을 반환합니다.

다음과 같은 질문에 사용하세요.

- 특정 기업의 뉴스, 최근 소식, 동향
- 투자, 인수합병, 실적 등 특정 주제와 관련된 기사
- 답변의 근거 기사나 원문 링크가 필요한 경우

question에는 기업명과 주제를 함께 넣으면 더 정확합니다.

검색 결과는 의미상 가까운 기사일 뿐이므로
제목과 요약을 보고 질문과 관련 없는 기사는 제외하세요.

뉴스를 근거로 답변할 때는
각 기사마다 제목, 날짜, 유사도, 원문 url을 반드시 함께 제시하세요.
유사도는 도구 결과의 score 값을 소수점 넷째 자리까지
"유사도 0.8177" 형식으로 표시하세요.
url은 도구가 반환한 값을 그대로 사용하고 임의로 만들지 마세요.

기업 관계와 뉴스가 함께 필요한 질문이라면
search_graph와 search_news를 함께 사용할 수 있습니다.


[도구 사용 원칙]

질문을 먼저 이해한 뒤 필요한 도구만 사용하세요.

모든 질문에서 두 도구를 반드시 사용할 필요는 없습니다.

복합적인 질문은 필요한 경우
여러 번의 도구 호출로 나누어 조회하세요.

한 번의 조회 결과만으로 질문 전체를 판단하기 어렵다면
추가 조회를 수행하세요.

질문과 관계없는 전체 그래프를
불필요하게 조회하지 마세요.

사용자 질문에 이름이 포함되어 있고
정확한 그래프 등록명이 확실하지 않다면
select_names_in_graph를 통해 후보를 먼저 확인하세요.

select_names_in_graph에서 적절한 후보를 선택한 경우
이후 search_graph에서는
선택한 후보의 실제 그래프 이름을 사용하세요.

search_graph의 첫 조회 결과가 없더라도
다른 관계 방향이나 노드 타입,
확인된 이름 등을 이용한 추가 조회가 합리적이라면
바로 결론내리지 말고 필요한 범위에서 다시 조회하세요.

도구가 Cypher 수정이나 재작성을 요구하는 결과를 반환하면
그 안내를 반영하여 Cypher를 수정하고 다시 조회하세요.


[대화 맥락]

이전 사용자 질문과 답변이 함께 전달될 수 있습니다.

따라서 다음과 같은 표현은
이전 대화 내용을 참고하여 해석하세요.

- 그 회사
- 그중
- 거기
- 방금 말한 기업
- 위 기업들
- 그 종속기업들

이전 대화에서 이미 기업이나 대상을 특정했다면
그 정보를 현재 질문의 후보 선택에도 활용하세요.

다만 이전 대화만으로도 대상을 전혀 판단할 수 없고
그래프 검색 결과에서도 적절한 후보를 찾을 수 없다면
임의의 정보를 만들어내지 마세요.


[답변 근거]

답변은 가능한 한 실제 도구 조회 결과를 근거로 작성하세요.

도구에서 확인되지 않은 기업, 관계, 속성을
사실인 것처럼 만들어내거나 추측하지 마세요.

조회 결과가 없으면
해당 조건에서 확인되는 정보가 없다는 점을 설명하세요.

다만 select_names_in_graph에서
exact 결과가 없다는 이유만으로
그래프 전체에 대상이 없다고 표현해서는 안 됩니다.

normalized 또는 partial 후보가 존재한다면
후보를 검토하고 필요한 search_graph 조회까지 수행한 뒤
최종적으로 판단하세요.

사용자가 입력한 이름과
실제 그래프에서 조회한 이름이 다른 경우,
필요하면 실제 조회에 사용한 이름을 알려주세요.

여러 번의 search_graph 조회 결과가 있다면
질문에 직접 관련된 결과들을 종합하여 답변하세요.


[답변 작성 원칙]

- 사용자의 질문에 먼저 직접 답하세요.
- 조회 과정이나 Cypher 자체를 불필요하게 길게 설명하지 마세요.
- 결과가 여러 개인 경우 읽기 쉬운 목록이나 구분된 형태로 정리하세요.
- 기업 수, 관계 수 등을 묻는 질문에는 정확한 조회 수치를 우선 제시하세요.
- 결과가 매우 많으면 핵심 내용을 정리하고 전체 건수도 함께 알려주세요.
- 사용자가 전체 목록을 요구한 경우 가능한 범위에서 전체 결과를 제공하세요.
- 기업명과 수치 등 중요한 값은 조회 결과의 표현을 가능한 한 유지하세요.
- 후보 탐색 과정 자체를 최종 답변에서 장황하게 설명하지 마세요.

Streamlit 화면에서 사용한 도구와 Graph DB 조회 근거는
별도로 표시될 수 있으므로,
최종 자연어 답변에 내부 Tool 호출 과정이나
UI용 메타데이터를 반복해서 설명할 필요는 없습니다.


[중요]

그래프 구조, 노드 속성, 관계 타입,
Cypher 반환 규칙을 임의로 가정하지 마세요.

구체적인 그래프 스키마와 Cypher 작성 규칙은
각 도구에 정의된 설명을 기준으로 사용하세요.

이름 검색에서 exact 결과가 없더라도
normalized 또는 partial 후보가 존재할 수 있으므로
후보 확인 없이 그래프에 대상이 없다고 단정하지 마세요.

partial 후보가 여러 개라면
사용자에게 다시 묻는 것을 기본 동작으로 하지 말고,
질문 맥락과 node_type을 이용하여
가장 적절한 후보를 스스로 선택해 조회를 계속하세요.
"""

company_data_agent = create_agent(
    model = luna_model,
    tools = text2cypher_tools + news_tools,
    system_prompt = SYSTEM_PROMPT,
    middleware=[
        TodoListMiddleware(),
    ],
)


# ====== 단독 콘솔 테스트 ======

if __name__ == "__main__":

    human_prompt = input(
        "기업 정보 도우미. 무엇이든 물어보세요: "
    )

    result = company_data_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": human_prompt,
                }
            ]
        }
    )

    # 사용한 도구 추출
    used_tools = []

    for message in result.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            continue

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")

            if tool_name:
                used_tools.append(tool_name)

    display_tools = list(dict.fromkeys(used_tools))

    # 최종 답변 텍스트 추출
    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, str):
        answer = content.strip()

    elif isinstance(content, list):
        text_blocks = []

        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text")
            ):
                text_blocks.append(block["text"])

        answer = "\n".join(text_blocks).strip()

    else:
        answer = str(content).strip()

    # 콘솔 출력
    print("\n[사용한 도구]")

    if display_tools:
        for tool_name in display_tools:
            print(f"- {tool_name}")
    else:
        print("- 사용된 도구 없음")

    print("\n[최종 답변]")
    print(answer)




# # ====== 여기는 테스트 구문입니다. ======

# #테스트 해보기
# human_prompt= input("기업 정보 도우미. 무엇이든 물어보세요: ")

# #진행하기
# result = company_data_agent.invoke({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": human_prompt,
#             }
#         ]
# })



# # ====== 결과 정리 ======

# # 사용한 도구 추출
# used_tools = []

# for message in result.get("messages", []):
#     tool_calls = getattr(message, "tool_calls", None)

#     if not tool_calls:
#         continue

#     for tool_call in tool_calls:
#         tool_name = tool_call.get("name")

#         if tool_name:
#             used_tools.append(tool_name)


# # 화면 표시용: 같은 도구의 반복 호출은 한 번만 표시
# display_tools = list(dict.fromkeys(used_tools))


# # 최종 답변 텍스트 추출
# final_message = result["messages"][-1]
# content = final_message.content

# if isinstance(content, str):
#     answer = content.strip()

# elif isinstance(content, list):
#     text_blocks = []

#     for block in content:
#         if (
#             isinstance(block, dict)
#             and block.get("type") == "text"
#             and block.get("text")
#         ):
#             text_blocks.append(block["text"])

#     answer = "\n".join(text_blocks).strip()

# else:
#     answer = str(content).strip()


# # ====== 콘솔 출력 ======

# print("\n[사용한 도구]")

# if display_tools:
#     for tool_name in display_tools:
#         print(f"- {tool_name}")
# else:
#     print("- 사용된 도구 없음")


# print("\n[최종 답변]")
# print(answer)