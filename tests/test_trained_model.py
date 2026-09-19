"""
`TrainedModel` loading: one shared instance per training config, and the
dynamic-input check that runs when a training config is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lstm import bmi_lstm

from helpers import BUNDLED_MODEL_CONFIG, GOLDEN_CONFIG, initialized


def test_members_of_one_config_share_one_model() -> None:
    """Two modules, and two members within one module, hold the same TrainedModel."""
    first = initialized(GOLDEN_CONFIG)
    second = initialized(GOLDEN_CONFIG)
    assert first.ensemble_members[0].model is second.ensemble_members[0].model

    model = first.ensemble_members[0].model
    assert model is bmi_lstm.load_trained_model(Path(BUNDLED_MODEL_CONFIG).resolve())
    assert not any(p.requires_grad for p in model.lstm.parameters())
    assert not model.lstm.training


def test_members_keep_their_own_state(nldas_forcing) -> None:
    """Sharing the model must not share the hidden and cell state between members."""
    stepped = initialized(GOLDEN_CONFIG)
    idle = initialized(GOLDEN_CONFIG)
    for name, value in nldas_forcing[0].items():
        stepped.set_value(name, value)
    stepped.update()
    assert stepped.ensemble_members[0].h_t.abs().sum() > 0
    assert idle.ensemble_members[0].h_t.abs().sum() == 0


def test_unmapped_dynamic_input_is_rejected_at_load(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path(BUNDLED_MODEL_CONFIG).read_text())
    cfg["dynamic_inputs"] = ["APCP_surface", "not_a_forcing"]
    bad = tmp_path / "config.yml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=r"Dynamic inputs \['not_a_forcing'\]"):
        bmi_lstm.load_trained_model(bad.resolve())
