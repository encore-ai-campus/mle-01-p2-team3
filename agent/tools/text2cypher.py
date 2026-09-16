import os
import json

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase

from pydantic import BaseModel, Field

# ====== 환경변수 / Neo4j 연결 ======

# 현재 파일에서 두 단계 상위 폴더의 .env
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

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

# 스키마 노드 타입 한정하기
EntityType = Literal[
    "ParentCompany","SubsidiaryCompany","Region","Industry",
]

# 노드, 관계 확인 함수
def select_names_in_graph(
    name: str,
    node_type: EntityType | None = None,
):
    # node_type이 지정된 경우 해당 노드 타입에서만 검색
    if node_type is not None:
        query = f"""
        MATCH (n:{node_type})
        WHERE n.name = $name
        RETURN
            n.name AS name,
            labels(n)[0] AS node_type,
            properties(n) AS properties
        """
    else:
        # 타입을 모르면 전체 노드에서 name으로 검색
        query = """
        MATCH (n)
        WHERE n.name = $name
        RETURN
            n.name AS name,
            labels(n)[0] AS node_type,
            properties(n) AS properties
        """

    records, summary, keys = driver.execute_query(
        query,
        name=name,
    )

    matches = [
        {
            "name": record["name"],
            "node_type": record["node_type"],
            "properties": record["properties"],
        }
        for record in records
    ]

    return {
        "query_name": name,
        "found": len(matches) > 0,
        "matches": matches,
    }


#도구1 독스트링
select_names_in_graph.__doc__ = f"""
질문에 등장한 기업명, 지역명, 업종명을
Neo4j에 실제 등록된 노드와 확인합니다.

이름(name)을 중심으로 검색하며,
node_type이 주어진 경우 해당 노드 타입으로 검색 범위를 제한합니다.

이 도구는 엔티티 존재 여부와 노드 타입을 확인하는 용도이며,
관계 탐색이나 복합 그래프 조회에는 사용하지 않습니다.

온톨로지:
{GRAPH_ONTOLOGY_TEXT}
"""


# ====== TOOL2 ======

# 입력 스키마
class GraphSearchInput(BaseModel):
    cypher: str = Field(
        description="Neo4j에서 실행할 조회 전용 Cypher 쿼리"
    )


# Cypher 실제 조회 함수
def search_graph(cypher: str):

    # 조회 이외의 쿼리 차단
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

    if any(keyword in upper_cypher for keyword in forbidden_keywords):
        raise ValueError(
            "search_graph는 조회 전용 도구입니다."
        )

    # Neo4j 조회
    records, summary, keys = driver.execute_query(cypher)

    # Record 객체를 일반 dict로 변환
    results = [
        dict(record)
        for record in records
    ]

    return {
        "cypher": cypher,
        "results": results,
    }


#도구2 독스트링
search_graph.__doc__ = f"""
Neo4j 그래프에서 관계와 속성을 조회합니다.

입력받은 Cypher를 검사한 뒤 Neo4j에서 실행하고,
조회 결과를 반환합니다.

반드시 조회 전용 Cypher를 사용해야 하며,
CREATE, MERGE, DELETE, SET 등의
데이터 변경 쿼리는 사용할 수 없습니다.

Cypher를 작성할 때는 아래 온톨로지에 정의된
노드, 관계, 속성을 사용하세요.

온톨로지:

{GRAPH_ONTOLOGY_TEXT}
"""