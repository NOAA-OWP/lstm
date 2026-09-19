"""
BMI config shapes for static attributes.

The module accepts two config shapes. The nested shape carries a
``static_attributes`` mapping. The flat shape has no such key and instead puts
every attribute named by the trained model config(s) at the top level, which is
the shape of the hydrofabric-generated JSON configs. Both shapes must build the
same module.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from lstm.bmi_lstm import bmi_LSTM

from helpers import GOLDEN_CONFIG, initialized, run, Forcing

STATIC = {"slope_mean": 9.95686, "elev_mean": 192.21}
AREA_SQKM = 427.77


def write_flat_yaml(path: Path, model_config: str, static: dict[str, float]) -> Path:
    lines = [f"train_cfg_file: {model_config}", f"area_sqkm: {AREA_SQKM}", "verbose: 0"]
    lines.extend(f"{name}: {value}" for name, value in static.items())
    path.write_text("\n".join(lines) + "\n")
    return path


def write_flat_json(path: Path, model_config: str, static: dict[str, float]) -> Path:
    """The hydrofabric-generated shape: flat keys, JSON syntax, a boolean `verbose`."""
    doc = {
        "area_sqkm": AREA_SQKM,
        **static,
        "train_cfg_file": model_config,
        "verbose": False,
    }
    path.write_text(json.dumps(doc, indent=3))
    return path


def static_values(model: bmi_LSTM) -> dict[str, float]:
    """The static attributes the module loaded, by name. They are internal, not BMI variables."""
    state = model._static_inputs
    return {name: float(state.value(name)[0]) for name in state.names()}


@pytest.mark.parametrize("write_flat", [write_flat_yaml, write_flat_json], ids=["yaml", "json"])
def test_flat_config_matches_nested_golden(
    write_flat, tmp_path: Path, bundled_model_config: str, nldas_forcing: Forcing
) -> None:
    flat_path = write_flat(tmp_path / "flat", bundled_model_config, STATIC)
    nested = initialized(GOLDEN_CONFIG)
    flat_model = initialized(flat_path)

    assert flat_model.get_input_var_names() == nested.get_input_var_names()
    assert static_values(flat_model) == static_values(nested)
    assert run(flat_model, nldas_forcing) == run(nested, nldas_forcing)


def test_flat_config_missing_attribute_names_it(tmp_path: Path, bundled_model_config: str) -> None:
    partial = {"slope_mean": STATIC["slope_mean"]}
    cfg = write_flat_yaml(tmp_path / "partial.yml", bundled_model_config, partial)
    with pytest.raises(ValueError, match=r"Missing static attributes: \['elev_mean'\]"):
        initialized(cfg)


def test_flat_config_covers_every_member(tmp_path: Path, bundled_model_config: str) -> None:
    """With two members the attribute names are the union over the members' training configs."""
    cfg = tmp_path / "two_member_flat.yml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            train_cfg_file:
              - {bundled_model_config}
              - {bundled_model_config}
            area_sqkm: {AREA_SQKM}
            slope_mean: {STATIC["slope_mean"]}
            elev_mean: {STATIC["elev_mean"]}
            """
        )
    )
    model = initialized(cfg)
    assert len(model.ensemble_members) == 2
    assert static_values(model) == STATIC


def test_nested_static_attributes_must_be_a_mapping(tmp_path: Path, bundled_model_config: str) -> None:
    cfg = tmp_path / "bad.yml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            train_cfg_file: {bundled_model_config}
            area_sqkm: {AREA_SQKM}
            static_attributes:
              - slope_mean
              - elev_mean
            """
        )
    )
    with pytest.raises(ValueError, match="'static_attributes' must be a mapping"):
        initialized(cfg)
