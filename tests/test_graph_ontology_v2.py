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

        self.assertEqual(set(section["properties"]), {"name", "category"})
        self.assertEqual(section["properties"]["name"]["type"], "string")
        self.assertEqual(section["properties"]["category"]["type"], "array[string]")

    def test_in_industry_supports_industry_and_section_targets(self):
        signatures = self.ontology["relationships"]["IN_INDUSTRY"]["signatures"]
        actual = {(row["source"], row["target"]) for row in signatures}
        expected = {
            ("ParentCompany", "Industry"),
            ("SubsidiaryCompany", "Industry"),
            ("ParentCompany", "Section"),
            ("SubsidiaryCompany", "Section"),
        }

        self.assertEqual(actual, expected)

    def test_relationship_signatures_reference_declared_nodes(self):
        node_labels = set(self.ontology["nodes"])

        for relationship in self.ontology["relationships"].values():
            for signature in relationship["signatures"]:
                self.assertIn(signature["source"], node_labels)
                self.assertIn(signature["target"], node_labels)


if __name__ == "__main__":
    unittest.main()
