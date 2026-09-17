import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from src.neo4j import prepare_graph_v2 as prep
from src.neo4j.load_graph import validate_graph


class PrepareGraphTest(unittest.TestCase):
    def test_deletion_uses_id_prefix_not_label(self):
        nodes = [
            {"id": "industry:remove", "type": "Section", "properties": {}},
            {"id": "keep", "type": "Industry", "properties": {}},
            {"id": "section:keep", "type": "Section", "properties": {}},
        ]
        kept, removed = prep.clean_v2(nodes)
        self.assertEqual([n["id"] for n in kept], ["keep", "section:keep"])
        self.assertEqual([n["id"] for n in removed], ["industry:remove"])

    def test_clean_does_not_mutate_input_and_drops_news(self):
        nodes = [
            {"id": "industry:x", "type": "Industry", "properties": {}},
            {"id": "news:1", "type": "News", "properties": {}},
            {"id": "section:금융", "type": "Section", "properties": {"name": "금융"}},
        ]
        before = copy.deepcopy(nodes)
        kept, _ = prep.clean_v2(nodes)
        self.assertEqual([n["id"] for n in kept], ["section:금융"])
        self.assertEqual(nodes, before)

    def test_news_uses_existing_vectors_and_rejects_missing_cache(self):
        article = {"record_id": "news:1", "company_crno": "0000000000001", "title": "t", "description": "s"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            vector = [0.1] * 768
            path.write_bytes(pickle.dumps({"t s": vector}))
            nodes = prep.news_from_cache([article], path)
            self.assertEqual(nodes[0]["properties"]["embedding"], vector)
            self.assertEqual(nodes[0]["properties"]["crno"], "0000000000001")
            path.write_bytes(pickle.dumps({}))
            with self.assertRaisesRegex(ValueError, "Missing/invalid"):
                prep.news_from_cache([article], path)

    def test_graph_section_relation_overrides_keyword_inference(self):
        nodes = [
            {"id": "parent:1", "type": "ParentCompany", "properties": {"sicNm": "전자"}},
            {"id": "section:금융", "type": "Section", "properties": {"name": "금융"}},
        ]
        triples = [{"subject": "parent:1", "object": "section:금융", "object_type": "Section"}]
        result = prep.graph_section_lookup(nodes, triples, lambda value: ["제조"])
        self.assertEqual(result["parent:1"], ["금융"])

    def test_csv_sections_match_graph_and_existing_cells_are_preserved(self):
        api = prep.notebook_api()
        tables = prep.classified_csvs(prep.CLEAN, api["classify_industry"])
        for name in [name for files in prep.CSV_FILES.values() for name in files]:
            fields, before = prep.read_csv(prep.CLEAN / name)
            _, after = tables[name]
            self.assertEqual(len(before), len(after))
            for old, new in zip(before, after):
                for field in fields:
                    self.assertEqual(old[field], new[field])
        overview = tables[prep.FINAL_CSV[prep.OVERVIEW]][1]
        lotte = next(r for r in overview if r["crno"] == "1101110000086")
        self.assertEqual(set(lotte["대분류"].split(";")), {"유통·물류"})

    def test_prepared_files_have_valid_endpoints_and_trailing_news_block(self):
        nodes = prep.read_jsonl(prep.CLEAN / prep.NODE_FILE_NAME)
        triples = prep.read_jsonl(prep.CLEAN / prep.TRIPLE_FILE_NAME)
        validate_graph(nodes, triples)
        articles = prep.read_jsonl(prep.CLEAN / prep.NEWS_FILE_NAME)
        news = nodes[-len(articles):]
        self.assertEqual([n["id"] for n in news], [a["record_id"] for a in articles])
        self.assertEqual(sum(n["type"] == "News" for n in nodes), len(articles))
        self.assertFalse(any(n["id"].startswith("industry:") for n in nodes))
        ontology = json.loads((prep.ROOT / "src/agent/tools/graph_ontology.json").read_text(encoding="utf-8"))
        for node in nodes:
            self.assertTrue(set(node["properties"]) <= set(ontology["nodes"][node["type"]]["properties"]))

    def test_loader_rejects_dangling_edges_before_connecting(self):
        row = {"subject": "missing", "subject_type": "News", "relation": "RELATED_TO",
               "object": "missing-company", "object_type": "ParentCompany"}
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate_graph([], [row])

    def test_crno_split_preserves_leading_zero_and_multiple_ids(self):
        self.assertEqual(prep.crnos("0000000000001;1101110000086"), ["0000000000001", "1101110000086"])


if __name__ == "__main__":
    unittest.main()
