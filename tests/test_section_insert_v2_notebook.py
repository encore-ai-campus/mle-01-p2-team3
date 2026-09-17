import json
import tempfile
import unittest
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "neo4j" / "section_insert_v2.ipynb"
)


def load_notebook_namespace():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "testable" in tags:
            exec("".join(cell["source"]), namespace)

    return namespace


class SectionInsertNotebookTest(unittest.TestCase):
    def test_industry_name_is_classified_by_section_keywords(self):
        namespace = load_notebook_namespace()
        classify_industry = namespace["classify_industry"]

        self.assertEqual(classify_industry("은행업"), ["금융"])
        self.assertEqual(classify_industry("자동차 부품 제조업"), ["제조"])
        self.assertEqual(
            classify_industry("그 외 기타 의료용 기기 제조업"),
            ["바이오·헬스케어"],
        )
        self.assertEqual(classify_industry("분류할 수 없는 업종"), ["모름"])

    def test_relationship_queries_connect_both_company_labels_to_section(self):
        namespace = load_notebook_namespace()

        self.assertIn(
            "MATCH (company:ParentCompany)-[:IN_INDUSTRY]->(industry:Industry)",
            namespace["PARENT_SECTION_RELATION_QUERY"],
        )
        self.assertIn(
            "MERGE (company)-[r:IN_INDUSTRY]->(section)",
            namespace["PARENT_SECTION_RELATION_QUERY"],
        )
        self.assertIn(
            "MATCH (company:SubsidiaryCompany)-[:IN_INDUSTRY]->(industry:Industry)",
            namespace["SUBSIDIARY_SECTION_RELATION_QUERY"],
        )
        self.assertIn(
            "MERGE (company)-[r:IN_INDUSTRY]->(section)",
            namespace["SUBSIDIARY_SECTION_RELATION_QUERY"],
        )

    def test_section_creation_is_idempotent_and_parameterized(self):
        namespace = load_notebook_namespace()
        query = namespace["SECTION_NODE_QUERY"]

        self.assertIn("UNWIND $sections AS row", query)
        self.assertIn("MERGE (section:Section {name: row.name})", query)
        self.assertIn("SET section.id = 'section:' + row.name", query)
        self.assertIn("SET section.category = row.category", query)

    def test_unknown_section_is_only_used_when_no_named_section_matches(self):
        namespace = load_notebook_namespace()

        for query_name in (
            "PARENT_UNKNOWN_RELATION_QUERY",
            "SUBSIDIARY_UNKNOWN_RELATION_QUERY",
        ):
            query = namespace[query_name]
            self.assertIn("NOT EXISTS", query)
            self.assertIn("Section {name: '모름'}", query)

    def test_build_section_nodes_uses_stable_ids(self):
        namespace = load_notebook_namespace()

        nodes = namespace["build_section_nodes"](
            [{"name": "금융", "category": ["은행"]}]
        )

        self.assertEqual(
            nodes,
            [
                {
                    "id": "section:금융",
                    "type": "Section",
                    "properties": {"name": "금융", "category": ["은행"]},
                }
            ],
        )

    def test_build_section_triples_assigns_unknown_only_when_company_has_no_match(self):
        namespace = load_notebook_namespace()
        nodes = [
            {
                "id": "industry:은행업",
                "type": "Industry",
                "properties": {"name": "은행업"},
            },
            {
                "id": "industry:불명",
                "type": "Industry",
                "properties": {"name": "분류할 수 없는 업종"},
            },
        ]
        triples = [
            {
                "subject": "parent_company:1",
                "subject_type": "ParentCompany",
                "relation": "IN_INDUSTRY",
                "object": "industry:은행업",
                "object_type": "Industry",
                "source_case": "1",
                "source_row": 10,
                "evidence": "은행업",
            },
            {
                "subject": "parent_company:1",
                "subject_type": "ParentCompany",
                "relation": "IN_INDUSTRY",
                "object": "industry:불명",
                "object_type": "Industry",
                "source_case": "1",
                "source_row": 11,
                "evidence": "분류할 수 없는 업종",
            },
            {
                "subject": "subsidiary_company:2",
                "subject_type": "SubsidiaryCompany",
                "relation": "IN_INDUSTRY",
                "object": "industry:불명",
                "object_type": "Industry",
                "source_case": "2",
                "source_row": 20,
                "evidence": "분류할 수 없는 업종",
            },
        ]

        section_triples = namespace["build_section_triples"](nodes, triples)
        endpoints = {
            (row["subject"], row["object"], row["object_type"])
            for row in section_triples
        }

        self.assertEqual(
            endpoints,
            {
                ("parent_company:1", "section:금융", "Section"),
                ("subsidiary_company:2", "section:모름", "Section"),
            },
        )

    def test_merge_graph_records_replaces_old_section_records_without_duplicates(self):
        namespace = load_notebook_namespace()
        old_section_node = {
            "id": "section:이전",
            "type": "Section",
            "properties": {"name": "이전", "category": []},
        }
        new_section_node = {
            "id": "section:금융",
            "type": "Section",
            "properties": {"name": "금융", "category": ["은행"]},
        }
        old_section_triple = {
            "subject": "parent_company:1",
            "subject_type": "ParentCompany",
            "relation": "IN_INDUSTRY",
            "object": "section:이전",
            "object_type": "Section",
        }
        industry_triple = {
            "subject": "parent_company:1",
            "subject_type": "ParentCompany",
            "relation": "IN_INDUSTRY",
            "object": "industry:은행업",
            "object_type": "Industry",
        }
        new_section_triple = {
            "subject": "parent_company:1",
            "subject_type": "ParentCompany",
            "relation": "IN_INDUSTRY",
            "object": "section:금융",
            "object_type": "Section",
        }

        merged_nodes, merged_triples = namespace["merge_graph_records"](
            [old_section_node],
            [old_section_triple, industry_triple],
            [new_section_node],
            [new_section_triple, new_section_triple],
        )

        self.assertEqual(merged_nodes, [new_section_node])
        self.assertEqual(merged_triples, [industry_triple, new_section_triple])

    def test_jsonl_round_trip_preserves_korean_text(self):
        namespace = load_notebook_namespace()
        rows = [{"id": "section:금융", "properties": {"name": "금융"}}]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "결과.jsonl"
            namespace["write_jsonl"](output_path, rows)

            self.assertEqual(namespace["read_jsonl"](output_path), rows)
            self.assertIn("금융", output_path.read_text(encoding="utf-8"))

    def test_validate_graph_records_rejects_missing_references(self):
        namespace = load_notebook_namespace()
        nodes = [
            {"id": "parent_company:1", "type": "ParentCompany", "properties": {}},
            {"id": "section:금융", "type": "Section", "properties": {}},
        ]
        valid_triple = {
            "subject": "parent_company:1",
            "subject_type": "ParentCompany",
            "relation": "IN_INDUSTRY",
            "object": "section:금융",
            "object_type": "Section",
        }

        summary = namespace["validate_graph_records"](nodes, [valid_triple])
        self.assertEqual(summary, {"node_count": 2, "triple_count": 1})

        invalid_triple = {**valid_triple, "object": "section:누락"}
        with self.assertRaisesRegex(ValueError, "참조할 수 없는 object"):
            namespace["validate_graph_records"](nodes, [invalid_triple])


if __name__ == "__main__":
    unittest.main()
