"""Verify Compose port selection without starting containers or resolving secrets."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is not installed")
@pytest.mark.parametrize("host_port", ["8000", "18001"])
def test_compose_host_port_override(host_port: str) -> None:
    """Render a distinct host port while preserving the application's container port."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env.example",
            "config",
            "--no-env-resolution",
            "--format",
            "json",
        ],
        cwd=root,
        env={**os.environ, "APP_PORT": host_port},
        capture_output=True,
        text=True,
        check=True,
    )
    ports = json.loads(result.stdout)["services"]["documind-app"]["ports"]
    assert len(ports) == 1
    assert str(ports[0]["published"]) == host_port
    assert ports[0]["target"] == 8000
