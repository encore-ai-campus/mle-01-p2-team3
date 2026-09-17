"""Load prepared v2 company/Section/News files: --target aura [--properties-only]."""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_FILE = PROJECT_ROOT / "data/clean/최종_기업관계_노드.jsonl"
TRIPLE_FILE = PROJECT_ROOT / "data/clean/최종_기업관계_트리플.jsonl"
VALID_LABELS = {"ParentCompany", "SubsidiaryCompany", "Region", "Section", "News"}
VALID_SIGNATURES = {
    ("ParentCompany", "AFFILIATED_WITH", "ParentCompany"),
    ("ParentCompany", "HAS_SUBSIDIARY", "SubsidiaryCompany"),
    ("ParentCompany", "LOCATED_IN", "Region"),
    ("SubsidiaryCompany", "LOCATED_IN", "Region"),
    ("ParentCompany", "IN_INDUSTRY", "Section"),
    ("SubsidiaryCompany", "IN_INDUSTRY", "Section"),
    ("News", "RELATED_TO", "ParentCompany"),
}
BATCH_SIZE = 250


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_graph(nodes, triples):
    by_id = {}
    for node in nodes:
        node_id = node["id"]
        if node_id in by_id:
            raise ValueError(f"Duplicate node ID: {node_id}")
        if node["type"] not in VALID_LABELS or node_id.startswith("industry:"):
            raise ValueError(f"Unexpected node type or industry ID: {node_id}")
        if "id" in node["properties"] and node["properties"]["id"] != node_id:
            raise ValueError(f"Conflicting property ID: {node_id}")
        for key, value in node["properties"].items():
            if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) or v is None for v in value)):
                raise ValueError(f"Unsupported Neo4j property: {node_id}.{key}")
        if node["type"] == "News":
            vector = node["properties"].get("embedding", [])
            if len(vector) != 768 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector) or not any(vector):
                raise ValueError(f"Invalid News embedding: {node_id}")
        by_id[node_id] = node
    seen = set()
    for row in triples:
        signature = (row["subject_type"], row["relation"], row["object_type"])
        if signature not in VALID_SIGNATURES:
            raise ValueError(f"Unsupported relationship: {signature}")
        key = (row["subject"], row["relation"], row["object"])
        if key in seen:
            raise ValueError(f"Duplicate relationship: {key}")
        seen.add(key)
        for role in ("subject", "object"):
            if row[role] not in by_id or by_id[row[role]]["type"] != row[f"{role}_type"]:
                raise ValueError(f"Invalid relationship endpoint: {row[role]}")
    return {"nodes": len(nodes), "relationships": len(triples)}


def create_constraints(session):
    for label in sorted(VALID_LABELS):
        session.run(f"CREATE CONSTRAINT graph_v2_{label}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()


def batches(rows):
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start:start + BATCH_SIZE]


def load_nodes(transaction, nodes, properties_only=False):
    groups = defaultdict(list)
    for node in nodes:
        if not properties_only or node["type"] in {"ParentCompany", "SubsidiaryCompany"}:
            groups[node["type"]].append(node)
    processed = {}
    for label, rows in groups.items():
        processed[label] = 0
        for batch in batches(rows):
            if properties_only:
                query = f"""UNWIND $rows AS row
                    MATCH (n:{label} {{id: row.id}})
                    FOREACH (k IN [key IN keys(row.properties)
                        WHERE n[key] IS NULL OR n[key] = '' OR n[key] = []] |
                        SET n[k] = row.properties[k])
                    RETURN count(n) AS count"""
            else:
                query = f"""UNWIND $rows AS row
                    MERGE (n:{label} {{id: row.id}})
                    SET n = row.properties, n.id = row.id
                    RETURN count(n) AS count"""
            count = transaction.run(query, rows=batch).single()["count"]
            if count != len(batch):
                raise ValueError(f"Missing or duplicate {label} targets; transaction rolled back")
            processed[label] += count
    return processed


def load_relationships(transaction, triples):
    groups = defaultdict(list)
    for row in triples:
        groups[(row["subject_type"], row["relation"], row["object_type"])].append(row)
    processed = 0
    for (source, relation, target), rows in groups.items():
        for batch in batches(rows):
            count = transaction.run(f"""UNWIND $rows AS row
                MATCH (a:{source} {{id: row.subject}}), (b:{target} {{id: row.object}})
                MERGE (a)-[r:{relation}]->(b)
                SET r.source_case = row.source_case,
                    r.source_row = row.source_row,
                    r.evidence = row.evidence
                RETURN count(r) AS count""", rows=batch).single()["count"]
            if count != len(batch):
                raise ValueError("Missing or duplicate relationship endpoints; transaction rolled back")
            processed += count
    return processed


def write_graph(transaction, nodes, triples, properties_only=False):
    counts = load_nodes(transaction, nodes, properties_only)
    relationships = 0 if properties_only else load_relationships(transaction, triples)
    return {"processed_nodes": counts, "processed_relationships": relationships}


def import_graph(target="aura", properties_only=False, require_empty=False,
                 node_file=NODE_FILE, triple_file=TRIPLE_FILE):
    nodes, triples = read_jsonl(node_file), read_jsonl(triple_file)
    validate_graph(nodes, triples)  # Validate before opening any database connection.
    load_dotenv(PROJECT_ROOT / ".env")
    if target not in {"aura", "local"}:
        raise ValueError("target must be aura or local")
    prefix = "AURA" if target == "aura" else "NEO4J"
    uri, user, password = (os.getenv(f"{prefix}_{key}") for key in ("URI", "USER", "PASSWORD"))
    if not all((uri, user, password)):
        raise RuntimeError(f"Set {prefix}_URI, {prefix}_USER and {prefix}_PASSWORD")
    database = os.getenv(f"{prefix}_DATABASE") or None
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            existing = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            if require_empty and existing:
                raise ValueError("Expected an empty database; no changes made")
            if properties_only and not existing:
                raise ValueError("Database is empty; import nodes before SET updates")
            if not properties_only:
                create_constraints(session)
            # Node/property/relationship writes commit or roll back together.
            result = session.execute_write(write_graph, nodes, triples, properties_only)
            if not properties_only and any(n["type"] == "News" for n in nodes):
                session.run("""CREATE VECTOR INDEX news_vec IF NOT EXISTS
                    FOR (n:News) ON n.embedding OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}""").consume()
                session.run("CALL db.awaitIndexes(120)").consume()
            result.update({
                "target": target,
                "node_counts": session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count").data(),
                "relationship_counts": session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count").data(),
                "industry_ids_remaining": session.run("MATCH (n) WHERE n.id STARTS WITH 'industry:' RETURN count(n) AS count").single()["count"],
                "news": session.run("MATCH (n:News) RETURN count(n) AS count, count(n.embedding) AS embedded, collect(DISTINCT size(n.embedding)) AS dimensions").single().data(),
                "vector_indexes": session.run("SHOW VECTOR INDEXES YIELD name, state, labelsOrTypes, properties RETURN name, state, labelsOrTypes, properties").data(),
            })
    report = PROJECT_ROOT / "data/clean" / f"graph_v2_{target}_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["aura", "local"], default=None)
    parser.add_argument("--properties-only", action="store_true")
    parser.add_argument("--require-empty", action="store_true")
    parser.add_argument("--nodes", type=Path, default=NODE_FILE)
    parser.add_argument("--triples", type=Path, default=TRIPLE_FILE)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    result = import_graph(args.target or os.getenv("GRAPH_DB", "local"), args.properties_only,
                          args.require_empty, args.nodes, args.triples)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
