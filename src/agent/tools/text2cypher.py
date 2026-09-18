import os
import json

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase

from pydantic import BaseModel, Field

# ====== 환경변수 / Neo4j 연결 ======

# 현재 파일에서 두 단계 상위 폴더의 .env
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# GRAPH_DB=aura 이면 로컬 대신 Aura(AURA_*) 의 그래프를 조회한다. 없으면 기존처럼 로컬.
GRAPH_DB = os.getenv("GRAPH_DB", "local").strip().lower()

if GRAPH_DB == "aura":
    from .news_vector import get_aura_driver

    driver = get_aura_driver()

else:
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise RuntimeError(
            "Neo4j 환경변수가 설정되지 않았습니다. "
            "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD를 확인하세요."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )


# ====== 온톨로지 ======

#온톨로지 경로 가져오기
ONTOLOGY_PATH = Path(__file__).resolve().parent / "graph_ontology.json"

#온톨로지 읽어오기
with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
    GRAPH_ONTOLOGY = json.load(f)

GRAPH_ONTOLOGY_TEXT = json.dumps(
    GRAPH_ONTOLOGY,
    ensure_ascii=False,
    indent=2,
)


# ====== TOOL1 ======

EntityType = Literal[
    "ParentCompany",
    "SubsidiaryCompany",
    "Region",
    "Section",
]


def normalize_entity_name(name: str) -> str:
    """
    기업명 비교용 정규화.
    법인 표기와 공백 차이를 제거한다.
    """
    normalized = name.strip()

    for token in [
        "(주)",
        "㈜",
        "주식회사",
    ]:
        normalized = normalized.replace(token, "")

    return normalized.replace(" ", "").lower()


def select_names_in_graph(
    name: str,
    node_type: EntityType | None = None,
):
    normalized_name = normalize_entity_name(name)

    if node_type is not None:
        query = f"""
        MATCH (n:{node_type})

        WITH
            n,
            toLower(
                replace(
                    replace(
                        replace(
                            replace(n.name, '(주)', ''),
                            '㈜', ''
                        ),
                        '주식회사', ''
                    ),
                    ' ', ''
                )
            ) AS normalized_node_name

        WHERE
            n.name = $name
            OR normalized_node_name = $normalized_name
            OR normalized_node_name CONTAINS $normalized_name

        RETURN
            n.name AS name,
            labels(n)[0] AS node_type,
            properties(n) AS properties,

            CASE
                WHEN n.name = $name THEN 'exact'
                WHEN normalized_node_name = $normalized_name
                    THEN 'normalized'
                ELSE 'partial'
            END AS match_type

        ORDER BY
            CASE
                WHEN n.name = $name THEN 0
                WHEN normalized_node_name = $normalized_name THEN 1
                ELSE 2
            END,
            size(n.name)

        LIMIT 20
        """

    else:
        query = """
        MATCH (n)

        WITH
            n,
            toLower(
                replace(
                    replace(
                        replace(
                            replace(n.name, '(주)', ''),
                            '㈜', ''
                        ),
                        '주식회사', ''
                    ),
                    ' ', ''
                )
            ) AS normalized_node_name

        WHERE
            n.name = $name
            OR normalized_node_name = $normalized_name
            OR normalized_node_name CONTAINS $normalized_name

        RETURN
            n.name AS name,
            labels(n)[0] AS node_type,
            properties(n) AS properties,

            CASE
                WHEN n.name = $name THEN 'exact'
                WHEN normalized_node_name = $normalized_name
                    THEN 'normalized'
                ELSE 'partial'
            END AS match_type

        ORDER BY
            CASE
                WHEN n.name = $name THEN 0
                WHEN normalized_node_name = $normalized_name THEN 1
                ELSE 2
            END,
            size(n.name)

        LIMIT 20
        """

    records, summary, keys = driver.execute_query(
        query,
        name=name,
        normalized_name=normalized_name,
    )

    matches = [
        {
            "name": record["name"],
            "node_type": record["node_type"],
            "properties": record["properties"],
            "match_type": record["match_type"],
        }
        for record in records
    ]

    return {
        "query_name": name,
        "normalized_query_name": normalized_name,
        "found": len(matches) > 0,
        "matches": matches,
    }


#도구1 독스트링
select_names_in_graph.__doc__ = f"""
질문에 등장한 기업명, 지역명, 업종명이
Neo4j에 어떤 이름으로 등록되어 있는지 확인합니다.

사용자가 입력한 이름과 그래프의 공식 이름은
'(주)', '㈜', '주식회사', 공백 등의 차이가 있을 수 있습니다.

검색 결과의 match_type은 다음 의미를 가집니다.

- exact:
  입력 이름과 그래프 이름이 정확히 일치합니다.

- normalized:
  법인 표기와 공백을 제거하면 동일한 이름입니다.
  일반적으로 가장 우선적으로 사용할 후보입니다.

- partial:
  입력 이름을 포함하는 관련 후보입니다.
  여러 후보가 존재할 수 있으므로 node_type과 실제 이름을
  확인한 뒤 적절한 노드를 선택해야 합니다.

정확히 일치하는 결과가 없더라도
normalized 또는 partial 후보가 있으면
그래프에 대상이 없다고 즉시 판단하지 마세요.

후보를 확인한 뒤 실제 관계 탐색은 search_graph를 사용하세요.

온톨로지:

{GRAPH_ONTOLOGY_TEXT}
"""


# ====== TOOL2 ======

# 입력 스키마
class GraphSearchInput(BaseModel):
    cypher: str = Field(
        description="Neo4j에서 실행할 조회 전용 Cypher 쿼리"
    )

# 그래프 근거 출력 규칙 

GRAPH_EVIDENCE_ALIASES = {
    "source",
    "relationship",
    "target",
}

AGGREGATE_FUNCTIONS = (
    "COUNT(",
    "SUM(",
    "AVG(",
    "MIN(",
    "MAX(",
    "COLLECT(",
)

def requires_graph_evidence_aliases(cypher: str) -> bool:
    """
    개별 관계를 조회하는 Cypher인지 확인합니다.

    단순 집계 조회는 그래프 시각화 대상에서 제외합니다.
    """

    upper_cypher = cypher.upper()

    # 관계 패턴 존재 여부
    has_relationship = (
        "-[" in upper_cypher
        and (
            "]->" in upper_cypher
            or "]-" in upper_cypher
        )
    )

    # COUNT, SUM 등의 집계 조회 여부
    is_aggregate = any(
        function in upper_cypher
        for function in AGGREGATE_FUNCTIONS
    )

    return has_relationship and not is_aggregate


def has_graph_evidence_aliases(cypher: str) -> bool:
    """
    관계 조회 결과에 UI가 필요한 표준 alias가 있는지 확인합니다.
    """

    upper_cypher = cypher.upper()

    return all(
        f" AS {alias.upper()}" in upper_cypher
        for alias in GRAPH_EVIDENCE_ALIASES
    )

# Cypher 실제 조회 함수
def search_graph(cypher: str):

    # ====== 조회 이외의 쿼리 차단 ======

    forbidden_keywords = [
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
    ]

    upper_cypher = cypher.upper()

    if any(
        keyword in upper_cypher
        for keyword in forbidden_keywords
    ):
        raise ValueError(
            "search_graph는 조회 전용 도구입니다."
        )


    # ====== 관계 조회 반환 형식 검사 ======

    if (
        requires_graph_evidence_aliases(cypher)
        and not has_graph_evidence_aliases(cypher)
    ):
        return {
            "cypher": cypher,
            "results": [],
            "error": (
                "개별 그래프 관계를 조회하는 경우 "
                "UI 근거 표시를 위해 RETURN 절에 "
                "source, relationship, target alias가 필요합니다. "
                "Cypher를 해당 형식으로 다시 작성해서 "
                "search_graph를 호출하세요."
            ),
            "required_aliases": [
                "source",
                "relationship",
                "target",
            ],
        }


    # ====== Neo4j 조회 ======

    records, summary, keys = driver.execute_query(
        cypher
    )


    # ====== Record → dict ======

    results = [
        dict(record)
        for record in records
    ]


    return {
        "cypher": cypher,
        "results": results,
    }


#도구2 독스트링
# ====== TOOL2 독스트링 ======

search_graph.__doc__ = f"""
Neo4j 그래프에서 관계와 속성을 조회합니다.

입력받은 Cypher를 검사한 뒤 Neo4j에서 실행하고
조회 결과를 반환합니다.

반드시 조회 전용 Cypher를 사용해야 하며,
CREATE, MERGE, DELETE, SET 등의
데이터 변경 쿼리는 사용할 수 없습니다.


[기본 조회 원칙]

사용자의 질문에서 기업, 계열사, 종속기업, 지역, 업종 등
그래프 관계로 표현되는 정보가 필요한 경우
노드 속성보다 그래프 관계를 우선적으로 확인하세요.

특히 기업의 업종, 산업, 분야, 대분류와 관련된 정보는
기업 노드의 업종 관련 속성을 직접 사용하는 것보다
IN_INDUSTRY 관계를 통해 연결된 Section 노드를
가장 우선적으로 조회하세요.


[업종 및 산업 조회 최우선 규칙]

사용자가 다음 의미의 정보를 요구하는 경우:

- 업종
- 산업
- 산업 분야
- 업종 분야
- 사업 분야
- 분야
- 대분류
- 어떤 업종에 속하는지
- 어떤 산업에 속하는지

기본 조회 대상은 반드시 다음 관계입니다.

ParentCompany -[:IN_INDUSTRY]-> Section

또는

SubsidiaryCompany -[:IN_INDUSTRY]-> Section

Section은 기업의 업종 대분류를 나타내는 노드이며,
Section.name이 해당 기업의 대분류명입니다.

따라서 업종 또는 산업 관련 질문에서는
다른 업종 관련 속성보다 Section.name을 우선적으로 사용하세요.

업종 관련 기본 우선순위는 다음과 같습니다.

1. IN_INDUSTRY 관계 존재 여부 확인
2. 연결된 Section 노드 확인
3. Section.name을 기업의 업종 대분류로 사용
4. 필요한 경우에만 추가 속성 조회

ParentCompany.sicNm은
원본 데이터에 저장된 개별 업종명입니다.

따라서 사용자가 단순히
"업종", "산업", "분야", "사업 분야"라고 질문한 경우에는
sicNm을 기본 업종 답변으로 사용하지 마세요.

sicNm은 다음과 같이
원본 또는 세부 업종 정보가 명시적으로 필요한 경우에만
조회하거나 답변에 사용하세요.

- 원본 업종명
- 기존 업종명
- 세부 업종
- sicNm

Section.category 역시
대분류를 판별하는 데 사용되는
소분류명 또는 키워드 목록이므로
사용자가 일반적인 업종이나 산업을 묻는 경우
Section.name 대신 반환하지 마세요.

즉 일반적인 업종 관련 질문의 기본값은 항상:

IN_INDUSTRY → Section.name

입니다.


[답변에서 업종 정보를 추가하는 경우]

사용자가 업종을 직접 질문하지 않았더라도
최종 답변에서 기업을 산업이나 업종별로 설명하거나
분야별로 분류하려는 경우에도
동일한 업종 조회 규칙을 적용하세요.

기업명, sicNm, business_content,
기업 이름에 포함된 단어,
모델이 알고 있는 일반 지식 등을 이용하여
임의로 기업의 산업 분야를 추론하지 마세요.

기업의 업종 또는 산업 분야를 답변에 포함하려면
반드시 해당 기업과 연결된
IN_INDUSTRY → Section 관계를 실제로 조회하세요.

IN_INDUSTRY 관계를 조회하지 않았다면
업종 또는 산업 분야를 추가하여 설명하지 마세요.

기업을 분야별로 묶거나 분류할 때도
Section.name을 기준으로 사용하세요.


[관계 조회 규칙]

기업-기업, 기업-지역, 기업-업종 등
개별 그래프 관계를 조회할 때는
UI에서 실제 Graph DB 근거를 표시할 수 있도록
RETURN 절에서 반드시 다음 alias를 사용하세요.

필수:
- source
- relationship
- target

가능하면 다음 정보도 함께 반환하세요.
- source_type
- target_type
- evidence
- source_case
- source_row


관계 조회의 권장 형식:

RETURN
    source_node.name AS source,
    labels(source_node)[0] AS source_type,
    type(rel) AS relationship,
    target_node.name AS target,
    labels(target_node)[0] AS target_type,
    rel.evidence AS evidence,
    rel.source_case AS source_case,
    rel.source_row AS source_row

source_node, rel, target_node은 예시 변수명이며
MATCH에서 사용한 변수명에 맞게 변경하세요.


[여러 홉 관계 조회 규칙]

두 개 이상의 관계를 거치는 질문은
경로 전체의 관계가 UI에 표시될 수 있도록
한 행에 관계 하나만 담아 반환하세요.

첫 번째 관계의 결과와
두 번째 관계의 최종값을 한 행의 별도 컬럼으로 합치지 마세요.

경로 전체를 path로 조회하고
relationships(path)를 UNWIND하여
각 관계마다 source, relationship, target을 반환하세요.

특히 다음과 같이
다른 기업을 거쳐 업종을 조회하는 질문에서는

ParentCompany
→ AFFILIATED_WITH 또는 HAS_SUBSIDIARY
→ 기업
→ IN_INDUSTRY
→ Section

의 전체 경로를 조회하세요.

최종 업종 대분류는
target_type이 Section인 관계 행의
target 값을 기준으로 판단하세요.


[질문 표현과 관계]

- 계열사, 계열회사, 같은 그룹 회사:
  AFFILIATED_WITH
  방향 없이 -[:AFFILIATED_WITH]- 로 조회

- 종속기업, 자회사:
  HAS_SUBSIDIARY
  ParentCompany → SubsidiaryCompany

- 업종, 산업, 분야, 사업 분야, 업종 분야, 대분류:
  IN_INDUSTRY
  ParentCompany 또는 SubsidiaryCompany → Section

- 지역, 위치:
  LOCATED_IN
  ParentCompany 또는 SubsidiaryCompany → Region

- 뉴스와 연결된 기업:
  RELATED_TO
  News → ParentCompany


[다른 기업을 거치는 질문]

"A의 계열사의 업종"
"A의 자회사들이 속한 산업"
"A 그룹 기업들의 분야"

처럼 다른 기업을 거쳐 업종이나 산업을 묻는 경우에는
A 자신의 업종 속성을 조회하지 마세요.

먼저 관련 기업을 관계로 찾은 뒤
각 기업에 대해 IN_INDUSTRY → Section을 조회하세요.

업종을 최종적으로 판단할 때는
각 기업의 Section.name을 사용하세요.


[속성 조회 규칙]

주소, 홈페이지, 대표자, 법인등록번호 등
노드 자체의 속성을 직접 묻는 경우에는
해당 노드의 속성을 조회할 수 있습니다.

그러나 다음 표현은
단순 속성 조회로 처리하지 마세요.

- 업종
- 산업
- 분야
- 사업 분야
- 업종 분야
- 대분류

이 표현들은 기본적으로
IN_INDUSTRY → Section 관계 조회로 처리하세요.


[집계 질문]

COUNT, SUM, AVG 등 집계 질문에서는
일반 관계 조회 반환 형식을 강제하지 않습니다.

다만 업종별 기업 수,
산업별 계열사 수,
대분류별 기업 수처럼
업종을 기준으로 집계하는 경우에는
반드시 Section을 기준으로 집계하세요.

sicNm이나 임의의 업종 문자열을 기준으로
대분류 집계를 대신하지 마세요.


[조회 범위]

질문에 필요한 범위만 조회하세요.

다만 업종이나 산업 정보를 답변에 사용하려면
필요한 IN_INDUSTRY → Section 조회는
생략하지 마세요.

조회 결과에 없는 업종이나 산업 정보를
추론하여 보완하지 마세요.


온톨로지:

{GRAPH_ONTOLOGY_TEXT}
"""