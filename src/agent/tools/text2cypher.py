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

두 개 이상의 관계를 거치는 질문(예: 계열사의 대분류,
종속기업의 지역)은 한 행에 관계 하나만 담기도록 반환하세요.
두 번째 홉을 industry_section 같은 별도 컬럼에 넣으면
UI 그래프에 첫 번째 홉만 표시됩니다.

경로 전체를 path 로 잡고 relationships(path) 를 UNWIND 해서
관계마다 한 행씩 source, relationship, target 을 반환하세요.

MATCH path = (p:ParentCompany {{name: '(주)한화'}})
             -[:AFFILIATED_WITH]-(a:ParentCompany)
             -[:IN_INDUSTRY]->(s:Section)
UNWIND relationships(path) AS rel
WITH DISTINCT rel, startNode(rel) AS source_node, endNode(rel) AS target_node
RETURN
    coalesce(source_node.name, source_node.title) AS source,
    labels(source_node)[0] AS source_type,
    type(rel) AS relationship,
    coalesce(target_node.name, target_node.title) AS target,
    labels(target_node)[0] AS target_type,
    rel.evidence AS evidence,
    rel.source_case AS source_case,
    rel.source_row AS source_row

답변은 이 행들을 종합해서 작성하세요.
(예: 계열사별로 target 이 Section 인 행을 모아 대분류를 정리)


[질문 표현과 관계]

- 계열사, 계열회사, 같은 그룹 회사: AFFILIATED_WITH (방향 없이 -[:AFFILIATED_WITH]- 로 조회)
- 종속기업, 자회사: HAS_SUBSIDIARY (ParentCompany → SubsidiaryCompany)
- 대분류, 업종 분야, 산업: IN_INDUSTRY → Section
- 지역, 위치: LOCATED_IN → Region
- 뉴스와 연결된 기업: RELATED_TO (News → ParentCompany)

"A 계열사의 ~" 처럼 다른 회사를 거쳐 묻는 질문은
A 자신의 속성이 아니라 연결된 회사들의 정보를 조회해야 합니다.


[예외]

단순 속성 조회에는 위 형식을 강제하지 않습니다.

COUNT, SUM, AVG 등의 집계 질문에도
위 형식을 강제하지 않습니다.

필요하지 않은 전체 그래프를 조회하지 말고
사용자 질문에 필요한 범위만 조회하세요.


온톨로지:

{GRAPH_ONTOLOGY_TEXT}
"""
