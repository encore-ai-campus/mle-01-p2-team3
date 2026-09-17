"""Final graph files (최종_*), cached News vectors, CSV sections and company properties.

Reads and writes only data/clean/최종_* files.

Offline: python src/neo4j/prepare_graph_v2.py
Aura property updates: add --apply-aura
Initial import into an empty Aura: add --apply-aura --initialize-empty
Classification comes directly from section_insert_v3.ipynb's testable cells.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import io
import json
import math
import pickle
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
COMPANIES = {"ParentCompany", "SubsidiaryCompany"}
# ParentCompany keeps only these properties; other overview columns are used for aliases only.
PARENT_PROPERTIES = ("crno", "name", "address", "aliases", "homepage", "representatives", "region", "sicNm")
SUBSIDIARY_PROPERTIES = ("name", "aliases", "address", "business_content", "domestic", "region")
NODE_PROPERTIES = {"ParentCompany": PARENT_PROPERTIES, "SubsidiaryCompany": SUBSIDIARY_PROPERTIES}
LABELS = COMPANIES | {"Region", "Section", "News"}
RELATIONS = {"AFFILIATED_WITH", "HAS_SUBSIDIARY", "LOCATED_IN", "IN_INDUSTRY", "RELATED_TO"}
OVERVIEW = "기업개요_최종"
SUBSIDIARY = "종속기업_정리"
INTEGRATED = "모기업_계열사_종속기업_통합"
# Final CSV per kind; these get the section columns.
CSV_FILES = {
    OVERVIEW: ("최종_기업개요.csv",),
    SUBSIDIARY: ("최종_종속기업_정리.csv",),
    INTEGRATED: ("최종_모기업_계열사_종속기업_통합.csv",),
}
FINAL_CSV = {kind: files[-1] for kind, files in CSV_FILES.items()}
CSV_KIND = {name: kind for kind, files in CSV_FILES.items() for name in files}
NODE_FILE_NAME = "최종_기업관계_노드.jsonl"
TRIPLE_FILE_NAME = "최종_기업관계_트리플.jsonl"
NEWS_FILE_NAME = "최종_뉴스기사.jsonl"


def notebook_api():
    namespace = {}
    notebook = json.loads((Path(__file__).with_name("section_insert_v3.ipynb")).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "testable" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return list(reader.fieldnames), rows


def text(value):
    if value is None or str(value).lower() in {"nan", "none", "nat"}:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value)).replace("&cr;", " ")).strip()


def normalize(value):
    return re.sub(r"[^0-9a-z가-힣]", "", text(value).lower())


def crnos(value):
    # Do not collapse multiple corporate IDs into a single invalid identifier.
    return list(dict.fromkeys(re.findall(r"(?<!\d)\d{13}(?!\d)", text(value))))


def subsidiary_id(row, index):
    name = normalize(row.get("name_norm")) or normalize(row.get("subsidiary_name"))
    address = normalize(row.get("subsidiary_addr"))
    domestic = normalize(row.get("domestic"))
    business = normalize(row.get("subsidiary_bizCtt"))
    if name and address:
        key = f"name:{name}|address:{address}"
    elif name and (domestic or business):
        key = f"name:{name}|fallback:{domestic}|{business}"
    elif name:
        key = f"name:{name}|row:{index}"
    else:
        key = f"address:{address}|row:{index}" if address else ""
    return f"subsidiary:{key}" if key else ""


def validate_nodes(nodes):
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate node IDs")


def clean_v2(v2):
    """Drop industry:* nodes (by ID prefix, irrespective of label) and News (re-added from articles)."""
    validate_nodes(v2)
    removed = [n for n in v2 if n["id"].startswith("industry:")]
    kept = [copy.deepcopy(n) for n in v2 if not n["id"].startswith("industry:") and n["type"] != "News"]
    return kept, removed


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Embedding cache must contain primitive values only")


def news_from_cache(articles, cache_path):
    cache = PrimitiveUnpickler(io.BytesIO(Path(cache_path).read_bytes())).load()
    nodes = []
    for article in articles:
        title = (article.get("title") or "").strip()
        summary = (article.get("description") or "").strip()
        vector = cache.get(f"{title} {summary}".strip())
        if not isinstance(vector, list) or len(vector) != 768 or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in vector
        ) or not any(vector):
            raise ValueError(f"Missing/invalid 768-dimensional vector: {article['record_id']}")
        try:
            date = parsedate_to_datetime(article.get("published_at")).date().isoformat()
        except (TypeError, ValueError):
            date = ""
        nodes.append({"id": article["record_id"], "type": "News", "properties": {
            "crno": article["company_crno"], "date": date, "title": title, "summary": summary,
            "publisher": (article.get("publisher") or "").strip(),
            "url": (article.get("original_url") or "").strip(), "embedding": vector,
        }})
    validate_nodes(nodes)
    return nodes


def graph_section_lookup(nodes, triples, classify):
    """Existing company -> Section links are authoritative, including multi-links."""
    by_id = {n["id"]: n for n in nodes}
    sections = defaultdict(set)
    for row in triples:
        if row["object_type"] == "Section" and row["object"] in by_id:
            sections[row["subject"]].add(by_id[row["object"]]["properties"]["name"])
    for node in nodes:
        if node["type"] in COMPANIES and not sections[node["id"]]:
            properties = node["properties"]
            if properties.get("section"):
                sections[node["id"]].update(properties["section"])
                continue
            industry = properties.get("sicNm") or properties.get("business_content") or ""
            if isinstance(industry, list):
                industry = ";".join(industry)
            sections[node["id"]].update(classify(industry))
    order = {n["properties"]["name"]: i for i, n in enumerate(nodes) if n["type"] == "Section"}
    return {key: sorted(values, key=lambda x: order.get(x, 999)) for key, values in sections.items()}


def classified_csvs(clean, classify, nodes=None, triples=None):
    """Keep every original cell/row; append only the requested section columns."""
    tables = {}
    for files in CSV_FILES.values():
        for name in files:
            tables[name] = read_csv(clean / name)
    nodes = nodes if nodes is not None else read_jsonl(clean / NODE_FILE_NAME)
    triples = triples if triples is not None else read_jsonl(clean / TRIPLE_FILE_NAME)
    sections = graph_section_lookup(nodes, triples, classify)
    by_id = {node["id"]: node for node in nodes}
    by_name = defaultdict(set)
    children = defaultdict(set)
    for node in nodes:
        if node["type"] == "SubsidiaryCompany":
            props = node["properties"]
            # name_norm is no longer a node property; the ID still carries it (subsidiary:name:<norm>|...).
            id_name = node["id"].split("name:", 1)[1].split("|", 1)[0] if "name:" in node["id"] else ""
            by_name[normalize(props.get("name_norm") or id_name or props.get("name"))].add(node["id"])
    for row in triples:
        if row["relation"] == "HAS_SUBSIDIARY":
            children[row["subject"]].add(row["object"])

    def section_text(ids):
        names = list(dict.fromkeys(s for key in ids for s in sections.get(key, ["모름"])))
        return ";".join(names) if names else "모름"

    def find_subsidiary(name, parent_ids, exact_id=None):
        if exact_id in by_id:
            return [exact_id]
        candidates = by_name.get(normalize(name), set())
        linked = set().union(*(children.get(f"parent_company:{p}", set()) for p in parent_ids))
        matches = candidates & linked
        if matches:
            return sorted(matches)
        return sorted(candidates) if len(candidates) == 1 else []

    for name, (fields, rows) in tables.items():
        if CSV_KIND[name] == OVERVIEW:
            new_fields = ["대분류"]
            for row in rows:
                row["대분류"] = section_text([f"parent_company:{row['crno']}"])
        elif CSV_KIND[name] == SUBSIDIARY:
            new_fields = ["대분류"]
            for row in rows:
                matches = find_subsidiary(row.get("name_norm") or row["sbrdEnpNm"], crnos(row["crno"]))
                row["대분류"] = section_text(matches)
        else:
            new_fields = ["대분류", "top_대분류", "affiliate_대분류", "subsidiary_대분류"]
            for index, row in enumerate(rows):
                for prefix in ("top", "affiliate"):
                    ids = [f"parent_company:{p}" for p in crnos(row[f"{prefix}_crno"])]
                    row[f"{prefix}_대분류"] = section_text(ids) if ids else ""
                parent_ids = crnos(row["affiliate_crno"]) or crnos(row["top_crno"])
                matches = find_subsidiary(row.get("name_norm") or row["subsidiary_name"], parent_ids, subsidiary_id(row, index))
                row["subsidiary_대분류"] = section_text(matches) if text(row["subsidiary_name"]) else ""
                row["대분류"] = ";".join(dict.fromkeys(
                    s for prefix in ("top", "affiliate", "subsidiary")
                    for s in row[f"{prefix}_대분류"].split(";") if s
                ))
        fields.extend(field for field in new_fields if field not in fields)
    return tables


def useful(value):
    return value not in (None, "", [], "-", "nan", "None")


def put_missing(properties, key, value):
    if useful(value) and not useful(properties.get(key)):
        properties[key] = value


def enrich_nodes(nodes, tables, classify, sections=None):
    by_id = {n["id"]: n for n in nodes}
    overview = {r["crno"]: r for r in tables[FINAL_CSV[OVERVIEW]][1]}
    additions = defaultdict(lambda: defaultdict(list))
    unmatched = Counter()

    def collect(node_id, values):
        if node_id not in by_id:
            return
        for key, value in values.items():
            if useful(value) and value not in additions[node_id][key]:
                additions[node_id][key].append(value)

    # Original integration has the exact name/address/row identity used by v2.
    # Supplemented rows are used only when their full ID still matches.
    for filename in CSV_FILES[INTEGRATED]:
        for i, row in enumerate(tables[filename][1]):
            for prefix in ("top", "affiliate"):
                ids = crnos(row.get(f"{prefix}_crno"))
                if len(ids) != 1:
                    if ids:
                        unmatched["multi_company_rows_skipped"] += 1
                    continue
                collect(f"parent_company:{ids[0]}", {
                    "corpNm": row[f"{prefix}_corpNm"], "region": row[f"{prefix}_region"],
                    "addr": row[f"{prefix}_addr"], "sicNm": row[f"{prefix}_sicNm"],
                })
            if text(row.get("subsidiary_name")):
                node_id = subsidiary_id(row, i)
                if node_id not in by_id:
                    unmatched[f"{filename}:unmatched_subsidiary_rows"] += 1
                    continue
                collect(node_id, {
                    "subsidiary_name": row["subsidiary_name"], "region": row["subsidiary_region"],
                    "subsidiary_addr": row["subsidiary_addr"], "subsidiary_bizCtt": row["subsidiary_bizCtt"],
                    "domestic": row["domestic"], "name_norm": row["name_norm"],
                })

    updated = Counter()
    for node in nodes:
        if node["type"] not in COMPANIES:
            continue
        props = node["properties"]
        before = copy.deepcopy(props)
        aliases = [props.get("name")]
        if node["type"] == "ParentCompany":
            row = overview.get(props.get("crno"))
            if row is None:
                unmatched["parent_nodes_without_overview"] += 1
            else:
                for key, value in row.items():
                    if key.startswith("Unnamed:") or key == "대분류":
                        continue
                    if key == "enpRprFnm" and value:
                        value = json.loads(value) if value.lstrip().startswith("[") else [value]
                        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                            raise ValueError(f"Invalid representatives: {node['id']}")
                    put_missing(props, key, value)
                aliases.extend(row.get(k) for k in ("corpNm", "corpEnsnNm", "enpPbanCmpyNm"))
                put_missing(props, "representatives", props.get("enpRprFnm"))
                put_missing(props, "english_name", row.get("corpEnsnNm"))
                put_missing(props, "homepage", row.get("enpHmpgUrl"))
        for key, values in additions[node["id"]].items():
            put_missing(props, key, values[0] if len(values) == 1 else values)
            if key in {"corpNm", "subsidiary_name"}:
                aliases.extend(values)
        old_aliases = props.get("aliases", [])
        if isinstance(old_aliases, str):
            old_aliases = [old_aliases]
        props["aliases"] = list(dict.fromkeys(v for v in [*old_aliases, *aliases] if useful(v)))
        industry = props.get("sicNm") or props.get("business_content") or ""
        if isinstance(industry, list):
            industry = ";".join(industry)
        props["section"] = (sections or {}).get(node["id"], classify(industry))
        node["properties"] = props = {k: props[k] for k in NODE_PROPERTIES[node["type"]] if useful(props.get(k))}
        if before != props:
            updated[node["type"]] += 1
    return {"updated_nodes": dict(updated), "matching_notes": dict(unmatched)}


def merge_triples(triples, nodes, removed_ids, news):
    by_id = {node["id"]: node for node in nodes}
    kept, seen = [], set()
    removed_count = 0
    for row in triples:
        if row["subject"] in removed_ids or row["object"] in removed_ids:
            removed_count += 1
            continue
        key = (row["subject"], row["relation"], row["object"])
        if key not in seen:
            kept.append(row)
            seen.add(key)
    for n in news:
        target = f"parent_company:{n['properties']['crno']}"
        row = {"subject": n["id"], "subject_type": "News", "relation": "RELATED_TO",
               "object": target, "object_type": "ParentCompany", "source_case": "news_articles",
               "source_row": None, "evidence": n["properties"]["url"]}
        key = (row["subject"], row["relation"], target)
        if key not in seen:
            kept.append(row)
            seen.add(key)
    for row in kept:
        for role in ("subject", "object"):
            if row[role] not in by_id or by_id[row[role]]["type"] != row[f"{role}_type"]:
                raise ValueError(f"Dangling or incorrectly typed reference: {row[role]}")
    return kept, removed_count


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_files(root=ROOT):
    root = Path(root)
    clean = root / "data" / "clean"
    api = notebook_api()
    v2_path = clean / NODE_FILE_NAME
    triple_path = clean / TRIPLE_FILE_NAME
    v2 = read_jsonl(v2_path)
    company_nodes, removed = clean_v2(v2)
    news = news_from_cache(read_jsonl(clean / NEWS_FILE_NAME), root / "data" / "news_emb_cache.pkl")
    old_triples = read_jsonl(triple_path)
    sections = graph_section_lookup(v2, old_triples, api["classify_industry"])
    tables = classified_csvs(clean, api["classify_industry"], v2, old_triples)
    enrichment = enrich_nodes(company_nodes, tables, api["classify_industry"], sections)
    merged = company_nodes + news
    validate_nodes(merged)
    # Prefix filtering also cleans already dangling Industry references on reruns.
    removed_ids = {n["id"] for n in removed} | {
        r[k] for r in old_triples for k in ("subject", "object") if r[k].startswith("industry:")
    }
    triples, removed_triples = merge_triples(old_triples, merged, removed_ids, news)
    backup = root / "data" / "work" / "graph_v2_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    # Final CSVs are the cleaner versions, so they get the section columns too.
    output_tables = tables
    for path in [v2_path, triple_path, *(clean / name for name in output_tables)]:
        shutil.copy2(path, backup / path.name)
    api["write_jsonl"](v2_path, merged)
    api["write_jsonl"](triple_path, triples)
    for name, (fields, rows) in output_tables.items():
        write_csv(clean / name, fields, rows)
    report = {
        "delete_condition": "id.startswith('industry:')", "removed_nodes": len(removed),
        "removed_industry_triples": removed_triples,
        "vector_source": f"{NEWS_FILE_NAME} + existing news_emb_cache.pkl (no embedding API calls)",
        "news_nodes": len(news), "embedding_dimensions": 768,
        "node_counts": dict(Counter(n["type"] for n in merged)), "total_nodes": len(merged),
        "total_triples": len(triples), "dangling_references": 0,
        "csv_rows": {name: len(rows) for name, (_, rows) in output_tables.items()},
        "backup": str(backup.relative_to(root)), **enrichment,
    }
    (clean / "graph_v2_preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def apply_aura(root=ROOT, initialize_empty=False):
    try:
        from .load_graph import import_graph
    except ImportError:
        from load_graph import import_graph
    clean = Path(root) / "data" / "clean"
    return import_graph(target="aura", properties_only=not initialize_empty,
                        require_empty=initialize_empty,
                        node_file=clean / NODE_FILE_NAME,
                        triple_file=clean / TRIPLE_FILE_NAME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-aura", action="store_true")
    parser.add_argument("--initialize-empty", action="store_true")
    parser.add_argument("--aura-only", action="store_true", help="Use already prepared files")
    args = parser.parse_args()
    if not args.aura_only:
        print(json.dumps(prepare_files(), ensure_ascii=False, indent=2))
    if args.apply_aura:
        print(json.dumps(apply_aura(initialize_empty=args.initialize_empty), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
