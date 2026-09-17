import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise RuntimeError(
        "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD를 .env에서 확인하세요."
    )

NODE_FILE = PROJECT_ROOT / "data" / "clean" / "기업관계_노드.jsonl"
TRIPLE_FILE = PROJECT_ROOT / "data" / "clean" / "기업관계_트리플.jsonl"

CONSTRAINTS = {
    "parent_company_id": "ParentCompany",
    "subsidiary_company_id": "SubsidiaryCompany",
    "region_id": "Region",
    "industry_id": "Industry",
}

VALID_LABELS = set(CONSTRAINTS.values())

VALID_RELATIONS = {
    "AFFILIATED_WITH",
    "HAS_SUBSIDIARY",
    "LOCATED_IN",
    "IN_INDUSTRY",
}


def create_constraints(session):
    for constraint_name, label in CONSTRAINTS.items():
        session.run(
            f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.id IS UNIQUE
            """
        ).consume()


def load_nodes(session):
    nodes_by_type = defaultdict(list)

    with NODE_FILE.open(encoding="utf-8") as file:
        for line in file:
            node = json.loads(line)

            if node["type"] not in VALID_LABELS:
                raise ValueError(f"허용되지 않은 노드 라벨: {node['type']}")

            nodes_by_type[node["type"]].append(node)

    for label, rows in nodes_by_type.items():
        session.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{id: row.id}})
            SET n += row.properties
            """,
            rows=rows,
        ).consume()


def load_relationships(session):
    triples_by_relation = defaultdict(list)
    with TRIPLE_FILE.open(encoding="utf-8") as file:
        for line in file:
            triple = json.loads(line)

            relation = triple['relation']

            if relation not in VALID_RELATIONS:
                raise ValueError(f"허용되지 않은 관계 타입: {relation}")

            triples_by_relation[relation].append(triple)

    for relation, rows in triples_by_relation.items():
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (subject {{id: row.subject}})
            MATCH (object {{id: row.object}})
            MERGE (subject)-[r:{relation}]->(object)
            SET r.source_case = row.source_case,
                r.source_row = row.source_row,
                r.evidence = row.evidence
            """,
            rows=rows,
        ).consume()



def print_node_counts(session):
    result = session.run(
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY label
        """
    )

    print("\n[노드 적재 결과]")
    for record in result:
        print(record.data())


def print_relationship_counts(session):
    result = session.run(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relation, count(r) AS count
        ORDER BY relation
        """
    )

    print("\n[관계 적재 결과]")
    for record in result:
        print(record.data())


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print("Neo4j 연결 성공")

        with driver.session() as session:
            create_constraints(session)
            print("제약조건 생성 확인")

            load_nodes(session)
            print("노드 적재 완료")

            load_relationships(session)
            print("관계 적재 완료")

            print_node_counts(session)
            print_relationship_counts(session)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
