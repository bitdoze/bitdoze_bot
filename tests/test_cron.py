from __future__ import annotations

from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.cron import load_cron_config


def test_load_cron_config_normalizes_session_scope_aliases(tmp_path: Path) -> None:
    cron_path = tmp_path / "CRON.yaml"
    cron_path.write_text(
        """
enabled: true
jobs:
  - name: shared-scope
    cron: "* * * * *"
    message: "a"
    session_scope: shared
  - name: private-scope
    cron: "* * * * *"
    message: "b"
    session_scope: private
""".strip(),
        encoding="utf-8",
    )
    config = Config(
        data={"cron": {"path": str(cron_path)}},
        path=tmp_path / "config.yaml",
    )

    cron_cfg = load_cron_config(config)

    assert cron_cfg.jobs[0].session_scope == "main"
    assert cron_cfg.jobs[1].session_scope == "isolated"


def test_load_cron_config_invalid_session_scope_defaults_to_isolated(tmp_path: Path) -> None:
    cron_path = tmp_path / "CRON.yaml"
    cron_path.write_text(
        """
enabled: true
jobs:
  - name: invalid-scope
    cron: "* * * * *"
    message: "a"
    session_scope: unexpected
""".strip(),
        encoding="utf-8",
    )
    config = Config(
        data={"cron": {"path": str(cron_path)}},
        path=tmp_path / "config.yaml",
    )

    cron_cfg = load_cron_config(config)

    assert cron_cfg.jobs[0].session_scope == "isolated"
