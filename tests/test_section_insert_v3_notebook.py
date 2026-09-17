import json
import tempfile
import unittest
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "neo4j" / "section_insert_v3.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def load_testable_namespace():
    namespace = {}
    for cell in load_notebook()["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if "testable" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace


class SectionInsertV3NotebookTest(unittest.TestCase):
    def test_build_section_triples_uses_neo4j_assignment_rows(self):
        namespace = load_testable_namespace()
        assignments = [
            {
                "subject": "parent_company:1",
                "subject_type": "ParentCompany",
                "industry_names": ["은행업", "분류할 수 없는 업종"],
                "source_row": 10,
            },
            {
                "subject": "subsidiary_company:2",
                "subject_type": "SubsidiaryCompany",
                "industry_names": ["분류할 수 없는 업종"],
                "source_row": 20,
            },
        ]

        triples = namespace["build_section_triples"](assignments)

        self.assertEqual(
            [(row["subject"], row["object"]) for row in triples],
            [
                ("parent_company:1", "section:금융"),
                ("subsidiary_company:2", "section:모름"),
            ],
        )
        self.assertTrue(all(row["source_case"] == "section_insert_v3" for row in triples))

    def test_validate_section_overlay_allows_existing_company_references(self):
        namespace = load_testable_namespace()
        section_nodes = namespace["build_section_nodes"](
            [{"name": "금융", "category": ["은행"]}]
        )
        triples = [
            {
                "subject": "parent_company:1",
                "subject_type": "ParentCompany",
                "relation": "IN_INDUSTRY",
                "object": "section:금융",
                "object_type": "Section",
                "source_case": "section_insert_v3",
                "source_row": 10,
                "evidence": "은행업",
            }
        ]

        summary = namespace["validate_section_overlay"](section_nodes, triples)

        self.assertEqual(summary, {"section_node_count": 1, "section_triple_count": 1})

    def test_validate_section_overlay_rejects_unknown_section_target(self):
        namespace = load_testable_namespace()
        section_nodes = namespace["build_section_nodes"](
            [{"name": "금융", "category": ["은행"]}]
        )
        invalid_triple = {
            "subject": "parent_company:1",
            "subject_type": "ParentCompany",
            "relation": "IN_INDUSTRY",
            "object": "section:누락",
            "object_type": "Section",
            "source_case": "section_insert_v3",
            "source_row": None,
            "evidence": "미확인",
        }

        with self.assertRaisesRegex(ValueError, "Section object를 찾을 수 없습니다"):
            namespace["validate_section_overlay"](section_nodes, [invalid_triple])

    def test_testable_api_does_not_read_existing_jsonl(self):
        namespace = load_testable_namespace()

        self.assertNotIn("read_jsonl", namespace)

    def test_source_query_reads_company_industry_assignments_from_neo4j(self):
        namespace = load_testable_namespace()
        query = namespace["COMPANY_INDUSTRY_SOURCE_QUERY"]

        self.assertIn("(company)-[relation:IN_INDUSTRY]->(industry:Industry)", query)
        self.assertIn("company.id AS subject", query)
        self.assertIn("collect(DISTINCT industry.name) AS industry_names", query)

    def test_jsonl_writer_creates_standalone_v3_files(self):
        namespace = load_testable_namespace()
        rows = [{"id": "section:금융", "type": "Section"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "기업관계_노드_v3.jsonl"
            namespace["write_jsonl"](output, rows)

            loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded, rows)

    def test_notebook_has_no_stale_execution_errors(self):
        for cell in load_notebook()["cells"]:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
