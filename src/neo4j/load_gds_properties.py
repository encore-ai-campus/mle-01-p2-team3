"""Load local GDS PageRank and Leiden properties into an existing Neo4j graph."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPERTIES_FILE = PROJECT_ROOT / "data" / "quality" / "gds_node_properties.jsonl"
VALID_LABELS = {"ParentCompany", "SubsidiaryCompany", "Region", "Section"}
BATCH_SIZE = 250


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    seen = set()
    for row in rows:
        node_id = row.get("id")
        label = row.get("label")
        if not node_id:
            raise ValueError("Missing id in GDS property row")
        if node_id in seen:
            raise ValueError(f"Duplicate id in GDS property rows: {node_id}")
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label for GDS property row: {label}")
        pagerank = row.get("gdsPageRank")
        community = row.get("gdsCommunityId")
        if pagerank is not None and not isinstance(pagerank, (int, float)):
            raise ValueError(f"Invalid gdsPageRank for {node_id}")
        if community is not None and not isinstance(community, int):
            raise ValueError(f"Invalid gdsCommunityId for {node_id}")
        seen.add(node_id)
    return {"rows": len(rows)}


def batches(rows: list[dict[str, Any]]):
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start:start + BATCH_SIZE]


def apply_properties(transaction: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    processed = {}
    for label, label_rows in grouped.items():
        processed[label] = 0
        for batch in batches(label_rows):
            count = transaction.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{label} {{id: row.id}})
                SET n.gdsPageRank = row.gdsPageRank,
                    n.gdsCommunityId = row.gdsCommunityId
                RETURN count(n) AS count
                """,
                rows=batch,
            ).single()["count"]
            if count != len(batch):
                raise ValueError(f"Missing Aura nodes for {label}; no changes committed")
            processed[label] += count
    return {"processed_nodes": processed}


def load_properties(
    target: str = "aura",
    properties_file: Path = DEFAULT_PROPERTIES_FILE,
) -> dict[str, Any]:
    rows = read_jsonl(properties_file)
    validate_rows(rows)

    load_dotenv(PROJECT_ROOT / ".env")
    if target not in {"aura", "local"}:
        raise ValueError("target must be aura or local")
    prefix = "AURA" if target == "aura" else "NEO4J"
    uri = os.getenv(f"{prefix}_URI")
    user = os.getenv(f"{prefix}_USER")
    password = os.getenv(f"{prefix}_PASSWORD")
    database = os.getenv(f"{prefix}_DATABASE") or None
    if not all((uri, user, password)):
        raise RuntimeError(f"Set {prefix}_URI, {prefix}_USER and {prefix}_PASSWORD")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            result = session.execute_write(apply_properties, rows)
            result.update({
                "target": target,
                "source_file": str(properties_file),
                "total_rows": len(rows),
                "property_counts": session.run(
                    """
                    MATCH (n)
                    WHERE n.gdsPageRank IS NOT NULL OR n.gdsCommunityId IS NOT NULL
                    RETURN count(n) AS nodes,
                           count(n.gdsPageRank) AS pagerank,
                           count(n.gdsCommunityId) AS community
                    """
                ).single().data(),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["aura", "local"], default="aura")
    parser.add_argument("--properties-file", type=Path, default=DEFAULT_PROPERTIES_FILE)
    args = parser.parse_args()
    print(json.dumps(load_properties(args.target, args.properties_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
