"""Geocode company addresses with Kakao Local and persist coordinates.

Usage:
    uv run python src/neo4j/geocode_parent_companies.py --target aura
    uv run python src/neo4j/geocode_parent_companies.py --target aura --entity subsidiary
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/address.json"


def clean_address(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("&cr;", " ").replace("\\n", " ").replace("\n", " ")
    return " ".join(text.split())


def normalize_address(value: str) -> str:
    return " ".join(clean_address(value).casefold().split())


def geocode(session: requests.Session, address: str, api_key: str) -> dict[str, Any]:
    response = session.get(
        KAKAO_URL,
        headers={"Authorization": f"KakaoAK {api_key}"},
        params={"query": address, "analyze_type": "similar", "size": 1},
        timeout=20,
    )
    if response.status_code == 429:
        time.sleep(1.0)
        response = session.get(
            KAKAO_URL,
            headers={"Authorization": f"KakaoAK {api_key}"},
            params={"query": address, "analyze_type": "similar", "size": 1},
            timeout=20,
        )
    response.raise_for_status()
    documents = response.json().get("documents", [])
    if not documents:
        return {"status": "no_match", "query_address": address}

    document = documents[0]
    return {
        "status": "ok",
        "query_address": address,
        "matched_address": document.get("address_name", ""),
        "address_type": document.get("address_type", ""),
        "longitude": float(document["x"]),
        "latitude": float(document["y"]),
    }


def query_companies(driver: Any, database: str | None, label: str) -> list[dict[str, Any]]:
    with driver.session(database=database) as session:
        return session.run(
            f"MATCH (p:{label}) "
            "RETURN p.id AS id, p.crno AS crno, p.name AS name, p.address AS address "
            "ORDER BY p.name"
        ).data()


def update_graph(driver: Any, database: str | None, rows: list[dict[str, Any]], label: str) -> int:
    payload = [
        {
            "id": row["id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "matched_address": row.get("matched_address", ""),
            "address_type": row.get("address_type", ""),
        }
        for row in rows
        if row.get("status") == "ok"
    ]
    if not payload:
        return 0

    with driver.session(database=database) as session:
        result = session.run(
            f"""
            UNWIND $rows AS row
            MATCH (p:{label} {{id: row.id}})
            SET p.latitude = row.latitude,
                p.longitude = row.longitude,
                p.geocoded_address = row.matched_address,
                p.geocode_address_type = row.address_type,
                p.geocode_provider = 'kakao_local'
            RETURN count(p) AS updated
            """,
            rows=payload,
        ).single()
    return int(result["updated"])


def write_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "crno",
        "name",
        "address",
        "status",
        "query_address",
        "matched_address",
        "address_type",
        "latitude",
        "longitude",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["aura", "local"], default="aura")
    parser.add_argument("--entity", choices=["parent", "subsidiary"], default="parent")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N companies (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between unique API requests")
    args = parser.parse_args()

    label = "ParentCompany" if args.entity == "parent" else "SubsidiaryCompany"
    output_file = PROJECT_ROOT / "data" / "clean" / f"{args.entity}_company_geocodes.csv"

    load_dotenv(PROJECT_ROOT / ".env")
    prefix = "AURA" if args.target == "aura" else "NEO4J"
    uri = os.getenv(f"{prefix}_URI")
    user = os.getenv(f"{prefix}_USER")
    password = os.getenv(f"{prefix}_PASSWORD")
    database = os.getenv(f"{prefix}_DATABASE") or None
    api_key = os.getenv("KAKAO_API_KEY")
    if not all((uri, user, password)):
        raise RuntimeError(f"Set {prefix}_URI, {prefix}_USER, {prefix}_PASSWORD in .env")
    if not api_key:
        raise RuntimeError("Set KAKAO_API_KEY in .env")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        companies = query_companies(driver, database, label)
        if args.limit > 0:
            companies = companies[: args.limit]

        http = requests.Session()
        cache: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        for index, company in enumerate(companies, start=1):
            address = clean_address(company.get("address"))
            row = {**company, "address": address}
            if not address:
                row["status"] = "missing_address"
            else:
                key = normalize_address(address)
                if key not in cache:
                    try:
                        cache[key] = geocode(http, address, api_key)
                    except requests.RequestException as exc:
                        cache[key] = {"status": f"request_error:{type(exc).__name__}"}
                    time.sleep(max(args.sleep, 0.0))
                row.update(cache[key])
            results.append(row)
            if index % 100 == 0 or index == len(companies):
                print(f"processed={index}/{len(companies)} ok={sum(r.get('status') == 'ok' for r in results)}", flush=True)

        write_csv(results, output_file)
        updated = update_graph(driver, database, results, label)
        summary = {
            "target": args.target,
            "entity": args.entity,
            "companies_processed": len(results),
            "unique_addresses": len(cache),
            "matched": sum(row.get("status") == "ok" for row in results),
            "missing_address": sum(row.get("status") == "missing_address" for row in results),
            "no_match": sum(row.get("status") == "no_match" for row in results),
            "updated_nodes": updated,
            "output": str(output_file),
        }
        print(summary)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
