"""Keep deterministic tests independent of the developer's private environment file."""

import pytest

from app.core.config import Settings


@pytest.fixture(autouse=True)
def isolate_settings_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never load local API credentials or deployment provider choices during tests."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
