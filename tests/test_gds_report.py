import inspect
import unittest

from src.neo4j import gds_report


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows

    def single(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        if "UNWIND $labels AS label" in query:
            return FakeResult([
                {"label": "ParentCompany", "count": 1},
                {"label": "SubsidiaryCompany", "count": 1},
                {"label": "Section", "count": 1},
                {"label": "Region", "count": 1},
            ])
        if "gds.graph.exists" in query:
            return FakeResult([{"exists": False}])
        if "gds.graph.project" in query:
            return FakeResult([{"graphName": "companyGraph", "nodeCount": 4, "relationshipCount": 3}])
        if "gds.pageRank.stream" in query:
            return FakeResult([
                {"rank": 1, "name": "A사", "label": "ParentCompany", "score": 2.5},
                {"rank": 2, "name": "B사", "label": "SubsidiaryCompany", "score": 1.2},
            ])
        if "gds.leiden.stream" in query:
            return FakeResult([
                {"rank": 1, "communityId": 7, "size": 3, "labels": ["ParentCompany"], "sampleNodes": ["A사", "B사"]},
                {"rank": 2, "communityId": 8, "size": 1, "labels": ["Section"], "sampleNodes": ["금융"]},
            ])
        raise AssertionError(f"Unexpected query: {query}")


class GdsReportTest(unittest.TestCase):
    def test_run_defaults_to_aura_target(self):
        self.assertEqual(gds_report.resolve_target(None), "local")
        self.assertIsNone(inspect.signature(gds_report.run).parameters["target"].default)

    def test_local_target_does_not_require_aura_gds_session(self):
        gds_report.ensure_aura_gds_config("local", {})

    def test_missing_required_labels_raise_actionable_error(self):
        class MissingLabelSession:
            def run(self, query, **parameters):
                return FakeResult([{"label": "ParentCompany", "count": 3}])

        with self.assertRaisesRegex(RuntimeError, "Missing required Neo4j labels: SubsidiaryCompany, Section, Region"):
            gds_report.ensure_required_labels(MissingLabelSession())

    def test_build_report_runs_pagerank_and_leiden_on_local_undirected_projection(self):
        session = FakeSession()

        report = gds_report.build_report(
            session,
            graph_name="companyGraph",
        )

        self.assertEqual(report["projection"]["nodeCount"], 4)
        self.assertEqual(report["pagerank_top10"][0]["name"], "A사")
        self.assertEqual(report["communities_top5"][0]["communityId"], 7)
        joined_queries = "\n".join(query for query, _ in session.queries)
        self.assertIn("gds.pageRank.stream", joined_queries)
        self.assertIn("gds.leiden.stream", joined_queries)
        project_parameters = next(parameters for query, parameters in session.queries if "gds.graph.project" in query)
        self.assertEqual(
            project_parameters["relationshipProjection"],
            {relationship: {"orientation": "UNDIRECTED"} for relationship in gds_report.RELATIONSHIPS},
        )
        self.assertEqual(project_parameters["configuration"], {})
        pagerank_parameters = next(parameters for query, parameters in session.queries if "gds.pageRank.stream" in query)
        leiden_parameters = next(parameters for query, parameters in session.queries if "gds.leiden.stream" in query)
        self.assertEqual(pagerank_parameters["configuration"], {})
        self.assertEqual(leiden_parameters["configuration"], {})

    def test_aura_projection_uses_relationship_type_list(self):
        self.assertEqual(gds_report.build_relationship_projection("aura"), gds_report.RELATIONSHIPS)

    def test_projection_config_uses_only_supported_keys(self):
        self.assertEqual(gds_report.build_projection_config({}), {})

    def test_resolve_gds_config_prefers_existing_session_id(self):
        config = gds_report.resolve_gds_config({"AURA_GDS_SESSION_ID": "existing-session"})

        self.assertEqual(config, {"sessionId": "existing-session"})

    def test_resolve_gds_config_can_create_aura_session_from_env(self):
        config = gds_report.resolve_gds_config({
            "AURA_GDS_MEMORY": "4GB",
            "AURA_GDS_PROVIDER": "aws",
            "AURA_GDS_REGION": "us-east-1",
            "AURA_GDS_PROJECT_ID": "project-1",
            "AURA_GDS_TTL": "PT1H",
        })

        self.assertEqual(config["memory"], "4GB")
        self.assertEqual(config["provider"], "aws")
        self.assertEqual(config["region"], "us-east-1")
        self.assertEqual(config["projectId"], "project-1")
        self.assertEqual(config["ttl"], "PT1H")

    def test_aura_requires_gds_session_or_creation_parameters(self):
        with self.assertRaisesRegex(RuntimeError, "AURA_GDS_SESSION_ID"):
            gds_report.ensure_aura_gds_config("aura", {})

    def test_markdown_report_contains_required_sections(self):
        report = {
            "graphName": "companyGraph",
            "projection": {"nodeCount": 4, "relationshipCount": 3},
            "pagerank_top10": [
                {"rank": 1, "name": "A사", "label": "ParentCompany", "score": 2.5}
            ],
            "communities_top5": [
                {"rank": 1, "communityId": 7, "size": 3, "labels": ["ParentCompany"], "sampleNodes": ["A사", "B사"]}
            ],
        }

        markdown = gds_report.to_markdown(report)

        self.assertIn("PageRank 허브 Top 10", markdown)
        self.assertIn("Leiden 커뮤니티 Top 5", markdown)
        self.assertIn("| 1 | A사 | ParentCompany | 2.500000 |", markdown)
        self.assertIn("A사, B사", markdown)


if __name__ == "__main__":
    unittest.main()
