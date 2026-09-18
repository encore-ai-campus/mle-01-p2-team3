from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "src" / "streamlit"
APP_PATH = APP_DIR / "app.py"


def load_app_module():
    sys.path.insert(0, str(APP_DIR))
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StreamlitAuraConfigTest(unittest.TestCase):
    def test_get_aura_config_prefers_streamlit_secrets_over_environment(self):
        app = load_app_module()

        config = app.get_aura_config(
            secrets={
                "AURA_URI": "neo4j+s://secret.databases.neo4j.io",
                "AURA_USER": "secret-user",
                "AURA_PASSWORD": "secret-password",
                "AURA_DATABASE": "secret-db",
            },
            environ={
                "AURA_URI": "neo4j+s://env.databases.neo4j.io",
                "AURA_USER": "env-user",
                "AURA_PASSWORD": "env-password",
                "AURA_DATABASE": "env-db",
            },
            load_env=False,
        )

        self.assertEqual(
            config,
            (
                "neo4j+s://secret.databases.neo4j.io",
                "secret-user",
                "secret-password",
                "secret-db",
            ),
        )

    def test_get_aura_config_falls_back_to_environment(self):
        app = load_app_module()

        config = app.get_aura_config(
            secrets={},
            environ={
                "AURA_URI": "neo4j+s://env.databases.neo4j.io",
                "AURA_USER": "env-user",
                "AURA_PASSWORD": "env-password",
                "AURA_DATABASE": "env-db",
            },
            load_env=False,
        )

        self.assertEqual(
            config,
            (
                "neo4j+s://env.databases.neo4j.io",
                "env-user",
                "env-password",
                "env-db",
            ),
        )


if __name__ == "__main__":
    unittest.main()
