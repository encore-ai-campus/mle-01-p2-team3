import unittest

from src.neo4j import load_gds_properties


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def single(self):
        return self.rows[0]


class FakeTransaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return FakeResult([{"count": len(parameters["rows"])}])


class LoadGdsPropertiesTest(unittest.TestCase):
    def test_validate_rows_accepts_supported_labels_and_properties(self):
        rows = [
            {
                "id": "parent:1",
                "label": "ParentCompany",
                "name": "A사",
                "gdsPageRank": 1.2,
                "gdsCommunityId": 3,
            }
        ]

        self.assertEqual(load_gds_properties.validate_rows(rows), {"rows": 1})

    def test_validate_rows_rejects_missing_id(self):
        with self.assertRaisesRegex(ValueError, "Missing id"):
            load_gds_properties.validate_rows([{"label": "ParentCompany"}])

    def test_apply_properties_matches_existing_nodes_by_id(self):
        rows = [
            {"id": "parent:1", "label": "ParentCompany", "gdsPageRank": 1.2, "gdsCommunityId": 3},
            {"id": "parent:2", "label": "ParentCompany", "gdsPageRank": 2.4, "gdsCommunityId": 4},
            {"id": "region:seoul", "label": "Region", "gdsPageRank": 5.0, "gdsCommunityId": 1},
        ]
        tx = FakeTransaction()

        result = load_gds_properties.apply_properties(tx, rows)

        self.assertEqual(result["processed_nodes"]["ParentCompany"], 2)
        self.assertEqual(result["processed_nodes"]["Region"], 1)
        query, parameters = tx.queries[0]
        self.assertIn("MATCH (n:ParentCompany {id: row.id})", query)
        self.assertIn("SET n.gdsPageRank", query)
        self.assertEqual(parameters["rows"][0]["id"], "parent:1")


if __name__ == "__main__":
    unittest.main()
