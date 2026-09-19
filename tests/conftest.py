"""
Shared fixtures for the ngen BMI Serialization Protocol tests.

Config files on this branch reference the bundled trained model relative to the
repository root, so every test runs with the repository root as its working
directory. The bundled forcing is read once per session and the dataset closed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from helpers import (
    BASIN_ID,
    BUNDLED_MODEL_CONFIG,
    FORCING_FILE,
    FORCING_VARIABLE_NAME_MAPPING,
    GOLDEN_CONFIG,
    REPO_ROOT,
    TOTAL_STEPS,
    Forcing,
)


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config files on this branch reference paths relative to the repository root."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture(scope="session")
def golden_config() -> Path:
    """The single-member config the golden integration test runs."""
    return GOLDEN_CONFIG


@pytest.fixture(scope="session")
def bundled_model_config() -> str:
    """The bundled trained model's config, as a repository-root-relative path."""
    return BUNDLED_MODEL_CONFIG


@pytest.fixture(scope="session")
def two_member_config(tmp_path_factory: pytest.TempPathFactory, bundled_model_config: str) -> Path:
    """
    A two-member ensemble config listing the bundled trained model twice, with
    the golden config's static attributes and area. The repository's example
    ensemble config references a second trained model that is not in the tree.
    """
    cfg = tmp_path_factory.mktemp("configs") / "two_member.yml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            train_cfg_file:
              - {bundled_model_config}
              - {bundled_model_config}
            basin_id: '{BASIN_ID}'
            basin_name: FALLING RIVER NEAR NARUNA, VA
            initial_state: zero
            time_step: 1 hour
            verbose: 0
            static_attributes:
              slope_mean: 9.95686
              elev_mean: 192.21
            area_sqkm: 427.77
            """
        )
    )
    return cfg


@pytest.fixture(params=[1, 2], ids=["single_member", "two_member"])
def member_count(request: pytest.FixtureRequest) -> int:
    """Parametrizes a test over the single-member and two-member configs."""
    return request.param


@pytest.fixture
def config(member_count: int, golden_config: Path, two_member_config: Path) -> Path:
    """The config with `member_count` members."""
    return {1: golden_config, 2: two_member_config}[member_count]


@pytest.fixture(scope="session")
def nldas_forcing() -> Forcing:
    """
    The first 24 hours of NLDAS forcing for the golden basin, as one dict per
    step keyed by BMI input name. Read once so every module sees identical values.
    """
    with nc.Dataset(FORCING_FILE, "r") as ds:
        basins = ds.variables["basin"][:]
        (basin_idxs,) = np.where(basins == BASIN_ID)
        assert len(basin_idxs) == 1
        basin_idx = basin_idxs[0]
        steps = []
        for ts in range(TOTAL_STEPS):
            steps.append(
                {
                    bmi_name: np.array(ds.variables[nc_name][basin_idx, ts], dtype="float64")
                    for nc_name, bmi_name in FORCING_VARIABLE_NAME_MAPPING.items()
                }
            )
    return steps
