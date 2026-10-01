from pathlib import Path

import pytest

from nl2sql.database import Database
from nl2sql.seed import seed_database
from nl2sql.semantic import SemanticLayer

ROOT = Path(__file__).resolve().parents[1]


def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return False
    return True


BACKENDS = ["sqlite", pytest.param("duckdb", marks=pytest.mark.skipif(not _has_duckdb(), reason="duckdb not installed"))]


@pytest.fixture(scope="session", params=BACKENDS)
def db_path(request, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp(request.param) / f"warehouse.{request.param}"
    seed_database(path)
    return path


@pytest.fixture()
def db(db_path):
    database = Database(db_path)
    yield database
    database.close()


@pytest.fixture(scope="session")
def semantic() -> SemanticLayer:
    return SemanticLayer.load(ROOT / "semantic_layer.yaml")


@pytest.fixture(scope="session")
def questions_path() -> Path:
    return ROOT / "evals" / "questions.yaml"
