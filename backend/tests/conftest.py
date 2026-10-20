import os
import tempfile
from pathlib import Path

# Point the app at a throwaway SQLite file *before* app.config is imported,
# so test runs never write scans/chat logs into the real backend/agri_system.db.
_TEST_DB = Path(tempfile.gettempdir()) / f"agri_system_test_{os.getpid()}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"


def pytest_sessionfinish(session, exitstatus):
    from app.db.session import engine

    engine.dispose()
    _TEST_DB.unlink(missing_ok=True)
