"""Create a Neo4j GDS PageRank and Leiden report for the company graph."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = PROJECT_ROOT / "data" / "quality" / "gds_report.json"
OUTPUT_MD = PROJECT_ROOT / "data" / "quality" / "gds_report.md"

NODE_LABELS = ["ParentCompany", "SubsidiaryCompany", "Section", "Region"]
RELATIONSHIPS = ["AFFILIATED_WITH", "HAS_SUBSIDIARY", "IN_INDUSTRY", "LOCATED_IN"]
GDS_SESSION_ENV_KEYS = ("AURA_GDS_SESSION_ID", "GDS_SESSION_ID")
GDS_CREATION_ENV_KEYS = {
    "memory": ("AURA_GDS_MEMORY", "GDS_MEMORY"),
    "provider": ("AURA_GDS_PROVIDER", "GDS_PROVIDER"),
    "region": ("AURA_GDS_REGION", "GDS_REGION"),
    "projectId": ("AURA_GDS_PROJECT_ID", "GDS_PROJECT_ID"),
    "ttl": ("AURA_GDS_TTL", "GDS_TTL"),
}


def resolve_target(value: str | None, default: str = "local") -> str:
    target = (value or default).strip().lower()
    if target not in {"aura", "local"}:
        raise ValueError("target must be aura or local")
    return target


def first_env_value(env: dict[str, str] | os._Environ[str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = (env.get(key) or "").strip()
        if value:
            return value
    return None


def resolve_gds_config(env: dict[str, str] | os._Environ[str] = os.environ) -> dict[str, Any]:
    session_id = first_env_value(env, GDS_SESSION_ENV_KEYS)
    if session_id:
        return {"sessionId": session_id}

    config: dict[str, Any] = {}
    for config_key, env_keys in GDS_CREATION_ENV_KEYS.items():
        value = first_env_value(env, env_keys)
        if value:
            config[config_key] = value
    return config


def build_projection_config(gds_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(gds_config or {})


def build_relationship_projection(target: str) -> Any:
    if target == "aura":
        return RELATIONSHIPS
    return {relationship: {"orientation": "UNDIRECTED"} for relationship in RELATIONSHIPS}


def ensure_aura_gds_config(target: str, gds_config: dict[str, Any]) -> None:
    if target != "aura":
        return
    if gds_config.get("sessionId") or gds_config.get("memory"):
        return
    raise RuntimeError(
        "Aura GDS requires graph analytics session settings. "
        "Set AURA_GDS_SESSION_ID if you already created a session, "
        "or set AURA_GDS_MEMORY with optional AURA_GDS_PROVIDER, AURA_GDS_REGION, "
        "AURA_GDS_PROJECT_ID and AURA_GDS_TTL so Aura can create one."
    )


def ensure_required_labels(session: Any) -> None:
    rows = session.run(
        """
        UNWIND $labels AS label
        CALL (label) {
          MATCH (n)
          WHERE label IN labels(n)
          RETURN count(n) AS count
        }
        RETURN label, count
        """,
        labels=NODE_LABELS,
    ).data()
    counts = {row["label"]: row["count"] for row in rows}
    missing = [label for label in NODE_LABELS if counts.get(label, 0) == 0]
    if missing:
        raise RuntimeError(
            "Missing required Neo4j labels: "
            + ", ".join(missing)
            + ". Load the final graph first or point --target to the database that already contains the final graph."
        )


def drop_graph_if_exists(session: Any, graph_name: str) -> bool:
    exists = session.run(
        "CALL gds.graph.exists($graphName) YIELD exists RETURN exists",
        graphName=graph_name,
    ).single()["exists"]
    if exists:
        session.run(
            "CALL gds.graph.drop($graphName) YIELD graphName RETURN graphName",
            graphName=graph_name,
        ).consume()
    return bool(exists)


def project_graph(
    session: Any,
    graph_name: str,
    gds_config: dict[str, Any] | None = None,
    target: str = "local",
) -> dict[str, Any]:
    drop_graph_if_exists(session, graph_name)
    return session.run(
        """
        CALL gds.graph.project(
          $graphName,
          $nodeLabels,
          $relationshipProjection,
          $configuration
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        graphName=graph_name,
        nodeLabels=NODE_LABELS,
        relationshipProjection=build_relationship_projection(target),
        configuration=build_projection_config(gds_config),
    ).single()


def run_pagerank(
    session: Any,
    graph_name: str,
    limit: int = 10,
    gds_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = session.run(
        """
        CALL gds.pageRank.stream($graphName, $configuration)
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN
          coalesce(node.name, node.title, node.id) AS name,
          labels(node)[0] AS label,
          score
        ORDER BY score DESC, name ASC
        LIMIT $limit
        """,
        graphName=graph_name,
        limit=limit,
        configuration=dict(gds_config or {}),
    ).data()
    return [
        {
            "rank": index,
            "name": row["name"],
            "label": row["label"],
            "score": float(row["score"]),
        }
        for index, row in enumerate(rows, start=1)
    ]


def run_leiden(
    session: Any,
    graph_name: str,
    limit: int = 5,
    gds_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = session.run(
        """
        CALL gds.leiden.stream($graphName, $configuration)
        YIELD nodeId, communityId
        WITH communityId, collect(gds.util.asNode(nodeId)) AS nodes
        WITH communityId, nodes, size(nodes) AS size
        ORDER BY size DESC
        LIMIT $limit
        UNWIND nodes AS node
        WITH communityId, size, nodes, labels(node)[0] AS label, count(*) AS labelCount
        ORDER BY communityId, labelCount DESC, label ASC
        WITH communityId, size, nodes, collect(label)[0..5] AS labels
        RETURN
          communityId,
          size,
          labels,
          [node IN nodes[0..10] | coalesce(node.name, node.title, node.id)] AS sampleNodes
        ORDER BY size DESC, communityId ASC
        """,
        graphName=graph_name,
        limit=limit,
        configuration=dict(gds_config or {}),
    ).data()
    return [
        {
            "rank": index,
            "communityId": int(row["communityId"]),
            "size": int(row["size"]),
            "labels": list(row.get("labels") or []),
            "sampleNodes": list(row.get("sampleNodes") or []),
        }
        for index, row in enumerate(rows, start=1)
    ]


def build_report(
    session: Any,
    graph_name: str = "companyGraph",
    gds_config: dict[str, Any] | None = None,
    target: str = "local",
) -> dict[str, Any]:
    ensure_required_labels(session)
    projection = dict(project_graph(session, graph_name, gds_config, target))
    return {
        "graphName": graph_name,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "projection": projection,
        "nodeLabels": NODE_LABELS,
        "relationships": RELATIONSHIPS,
        "pagerank_top10": run_pagerank(session, graph_name, gds_config=gds_config),
        "communities_top5": run_leiden(session, graph_name, gds_config=gds_config),
    }


def to_markdown(report: dict[str, Any]) -> str:
    projection = report["projection"]
    pagerank_rows = "\n".join(
        f"| {row['rank']} | {row['name']} | {row['label']} | {row['score']:.6f} |"
        for row in report["pagerank_top10"]
    )
    community_rows = "\n".join(
        "| {rank} | {communityId} | {size} | {labels} | {sampleNodes} |".format(
            rank=row["rank"],
            communityId=row["communityId"],
            size=row["size"],
            labels=", ".join(row["labels"]),
            sampleNodes=", ".join(row["sampleNodes"]),
        )
        for row in report["communities_top5"]
    )
    return f"""# GDS 그래프 분석 리포트

## 분석 기준

| 항목 | 값 |
| --- | ---: |
| GDS 그래프명 | `{report["graphName"]}` |
| 노드 수 | {projection["nodeCount"]} |
| 관계 수 | {projection["relationshipCount"]} |

## PageRank 허브 Top 10

| 순위 | 노드 | 라벨 | PageRank |
| ---: | --- | --- | ---: |
{pagerank_rows}

## Leiden 커뮤니티 Top 5

| 순위 | 커뮤니티 ID | 노드 수 | 주요 라벨 | 대표 노드 |
| ---: | ---: | ---: | --- | --- |
{community_rows}

## 해석 기준

- PageRank는 중요한 노드와 연결된 노드를 더 높게 평가하는 중심성 지표입니다.
- Leiden 커뮤니티는 서로 촘촘히 연결된 노드 묶음을 찾는 커뮤니티 탐지 결과입니다.
- 이 리포트는 `ParentCompany`, `SubsidiaryCompany`, `Section`, `Region`과 기업 관계만 대상으로 하며, 뉴스 노드는 검색 근거용이라 제외했습니다.
"""


def write_report(report: dict[str, Any], json_path: Path = OUTPUT_JSON, md_path: Path = OUTPUT_MD) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")


def run(
    target: str | None = None,
    graph_name: str = "companyGraph",
    gds_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    target = resolve_target(target or os.getenv("GRAPH_DB"))
    gds_config = dict(gds_config if gds_config is not None else resolve_gds_config())
    ensure_aura_gds_config(target, gds_config)
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
            report = build_report(session, graph_name, gds_config, target)
    write_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["local", "aura"], default=resolve_target(os.getenv("GRAPH_DB")))
    parser.add_argument("--graph-name", default="companyGraph")
    parser.add_argument("--gds-session-id")
    parser.add_argument("--gds-memory")
    parser.add_argument("--gds-provider")
    parser.add_argument("--gds-region")
    parser.add_argument("--gds-project-id")
    parser.add_argument("--gds-ttl")
    args = parser.parse_args()
    gds_config = resolve_gds_config()
    if args.gds_session_id:
        gds_config = {"sessionId": args.gds_session_id}
    else:
        for config_key, arg_value in {
            "memory": args.gds_memory,
            "provider": args.gds_provider,
            "region": args.gds_region,
            "projectId": args.gds_project_id,
            "ttl": args.gds_ttl,
        }.items():
            if arg_value:
                gds_config[config_key] = arg_value
    print(json.dumps(run(args.target, args.graph_name, gds_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
