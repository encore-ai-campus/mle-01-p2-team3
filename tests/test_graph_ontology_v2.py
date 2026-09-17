import json
import unittest
from pathlib import Path


ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agent"
    / "tools"
    / "graph_ontology_v2.json"
)


class GraphOntologyV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))

    def test_section_node_exposes_name_and_category_properties(self):
        section = self.ontology["nodes"]["Section"]

        self.assertEqual(set(section["properties"]), {"id", "name", "category"})
        self.assertEqual(section["properties"]["name"]["type"], "string")
        self.assertEqual(section["properties"]["category"]["type"], "array[string]")

    def test_in_industry_supports_only_section_targets(self):
        signatures = self.ontology["relationships"]["IN_INDUSTRY"]["signatures"]
        actual = {(row["source"], row["target"]) for row in signatures}
        expected = {
            ("ParentCompany", "Section"),
            ("SubsidiaryCompany", "Section"),
        }

        self.assertEqual(actual, expected)
        self.assertNotIn("Industry", self.ontology["nodes"])

    def test_runtime_ontology_matches_v2_and_exposes_news_and_company_details(self):
        runtime = json.loads(ONTOLOGY_PATH.with_name("graph_ontology.json").read_text(encoding="utf-8"))
        self.assertEqual(runtime, self.ontology)
        self.assertIn("News", runtime["nodes"])
        self.assertIn("RELATED_TO", runtime["relationships"])
        self.assertIn("representatives", runtime["nodes"]["ParentCompany"]["properties"])
        self.assertIn("aliases", runtime["nodes"]["ParentCompany"]["properties"])

    def test_relationship_signatures_reference_declared_nodes(self):
        node_labels = set(self.ontology["nodes"])

        for relationship in self.ontology["relationships"].values():
            for signature in relationship["signatures"]:
                self.assertIn(signature["source"], node_labels)
                self.assertIn(signature["target"], node_labels)


if __name__ == "__main__":
    unittest.main()
