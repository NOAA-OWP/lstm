"""
Golden-bytes tests that lock the serialization payload format.

The fixed-snapshot literal pins the exact bytes of format version 1: its values
are hand-chosen and no arithmetic sits between them and the bytes, so it must
match on every platform. A failure there means the payload format changed,
which requires a format version bump and an explicit decision, not a new
literal.

The real-module literal was captured from the bundled model on one machine.
Its header, fingerprint, and member header are exact everywhere, but its
hidden, cell, and output sections are torch results whose last bits differ
between CPU architectures and torch builds, so those sections are compared
with a tolerance. Same-machine bitwise faithfulness of the capture path is
covered by the capture-and-restore and end-to-end tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lstm import bmi_lstm
from lstm import serialization_codec as codec

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_CONFIG = REPO_ROOT / "configs/02064000_nh_NLDAS_hourly.yml"

TRIGGER = np.array([1], dtype="int32")


# --------------------------------------------------------------------------
# Fixed snapshot: two members of different hidden sizes, two outputs
# --------------------------------------------------------------------------

FIXED_FINGERPRINT = (
    b"lstm-bmi;members=2;"
    b"0:hidden=2,inputs=a|b,run=r,epoch=1;"
    b"1:hidden=3,inputs=a|b,run=s,epoch=2"
)


def _fixed_snapshot() -> codec.Snapshot:
    """A snapshot whose every value is exactly representable in its wire type."""
    return codec.Snapshot(
        timestep=3,
        fingerprint=FIXED_FINGERPRINT,
        members=[
            (
                np.array([1.0, -2.0], dtype=np.float32),
                np.array([0.5, 0.25], dtype=np.float32),
            ),
            (
                np.array([1.5, 2.5, 3.5], dtype=np.float32),
                np.array([-0.125, 0.0, 8.0], dtype=np.float32),
            ),
        ],
        outputs=np.array([0.125, -8.0], dtype=np.float64),
    )


# 186 bytes: 32-byte header, 90-byte fingerprint, member 0 (4 + 8 + 8),
# member 1 (4 + 12 + 12), and two float64 outputs (16).
FIXED_SNAPSHOT_PAYLOAD = bytes.fromhex(
    "4c53544d424d490001000000030000000000000002000000020000005a000000"
    "6c73746d2d626d693b6d656d626572733d323b303a68696464656e3d322c696e"
    "707574733d617c622c72756e3d722c65706f63683d313b313a68696464656e3d"
    "332c696e707574733d617c622c72756e3d732c65706f63683d32020000000000"
    "803f000000c00000003f0000803e030000000000c03f00002040000060400000"
    "00be0000000000000041000000000000c03f00000000000020c0"
)


def test_fixed_snapshot_packs_to_golden_bytes():
    assert codec.pack(_fixed_snapshot()) == FIXED_SNAPSHOT_PAYLOAD


def test_golden_bytes_unpack_to_fixed_snapshot():
    expected = _fixed_snapshot()
    restored = codec.unpack(FIXED_SNAPSHOT_PAYLOAD)
    assert restored.timestep == expected.timestep
    assert restored.fingerprint == expected.fingerprint
    assert len(restored.members) == len(expected.members)
    for (hidden, cell), (expected_hidden, expected_cell) in zip(
        restored.members, expected.members
    ):
        assert hidden.dtype == np.float32
        assert cell.dtype == np.float32
        assert np.array_equal(hidden, expected_hidden)
        assert np.array_equal(cell, expected_cell)
    assert restored.outputs.dtype == np.float64
    assert np.array_equal(restored.outputs, expected.outputs)


# --------------------------------------------------------------------------
# Real payload captured from the golden single-member module
# --------------------------------------------------------------------------

# Every BMI input of the golden config, with a per-input multiplier. At step
# ``s`` (1-based) each input is set to ``s * multiplier * 0.5``.
INPUT_MULTIPLIERS = {
    "land_surface_radiation~incoming~longwave__energy_flux": 1,
    "land_surface_air__pressure": 2,
    "atmosphere_air_water~vapor__relative_saturation": 3,
    "atmosphere_water__liquid_equivalent_precipitation_rate": 4,
    "land_surface_radiation~incoming~shortwave__energy_flux": 5,
    "land_surface_air__temperature": 6,
    "land_surface_wind__x_component_of_velocity": 7,
    "land_surface_wind__y_component_of_velocity": 8,
}
CAPTURE_STEPS = 3

# 1227 bytes: header, the golden model's fingerprint, one member with hidden
# size 126, and two float64 outputs, captured after three deterministic steps
# on an aarch64 Linux machine. The float sections are platform-dependent in
# their last bits; see the module docstring.
REAL_MODULE_PAYLOAD = bytes.fromhex(
    "4c53544d424d49000100000003000000000000000100000002000000a7000000"
    "6c73746d2d626d693b6d656d626572733d313b303a68696464656e3d3132362c"
    "696e707574733d415043505f737572666163657c544d505f326d61626f766567"
    "726f756e647c656c65765f6d65616e7c736c6f70655f6d65616e2c72756e3d6e"
    "685f414f52435f686f75726c795f736c6f70655f656c65765f7072656369705f"
    "74656d705f7365713939395f736565643130315f323830315f3139313830362c"
    "65706f63683d397e0000002131c0bc5d93b6b6880d5f3b9cd4aba685337b3fbd"
    "b77836114b95b9117512b8019856b7d49f5d3fb874d83107f1052ad46397be37"
    "40302bb6b045b6f2478d325d7281b4f3020e38e1d8beb4b145e4b948af17b529"
    "e092bc15ec7bbf39bf13b772c7723f982109badb9c113a70a8d73e787f9b3737"
    "e4a53bb4480a2fcf25653b1a42443964fd3c380019773f0264a6b97d3e113ab5"
    "ceeebea3f5a63aa27b35bfba324cbfc8bc91bba26d803ed29be6af4b408338fe"
    "aa5c37e52a94bb6b1fa5ab0ac32d3db5e9d8ba587c053755bb7e3f2f3ef4b8d0"
    "0c7e3f13391c3a88b8d6b98b58a828f93c3a39a3d418b5edda5db718e89daffb"
    "5996be7840f6b269a6b6b840ea2b35cc8abdb0b45f2d3f4370743f04761ebc0b"
    "8095b84d2e733f2cd2453b31af8b3e114ea0b4acb2253391a50a32cc4c983e3b"
    "2643bfec85f238da067d3d6acf8c301752e231407d283c1274f43c893893b3d0"
    "b2e8bc699679bf5a917abfb9437b3daf153ebed42348bd7c11543fc7f3adad26"
    "da83ae121d2fbf8fab7abb5081502fa4f3fcb84935163a977114badb9e821fc2"
    "c835afd9b827af400fc4b19b164e3f225e533b9e946cbf0bf24ebd63e3ab365c"
    "2a2b364f8a55b2cf1a473dc62e41b81135a0b77d0d8ebe25ef742e03336fbf96"
    "028831a7eaf639bddaac255d73a1b3f73075bf912b6eb7014f8839e8a55730c4"
    "e24fbff2c005bf3dd197b944f0b53f09a06fbf5a1c1640a651f13631b3cbbff5"
    "2aeeb8b90a98bd6b86a83fef81013446e60637608d30c0fffd123374b645b62b"
    "488d32978c88bf374ceb38b2a819bb9d46e4b9a8981eb5df2d77bf4f6e1ac027"
    "d93fc04259e83f97816ebb30a1113abb23e73eef689c3779eda53b7ca4122fad"
    "0d853b795d4a39f22041385f1a01402bbb3fbf5baf113a2ac14fbf2c89813ce2"
    "47b3bf4b1e8cbf464e9dbc6a45063f705a86b28e558538023c97386a9222c076"
    "1f88bc1eff2d3e44dac0be83a18e3b64f13f409462f4b8950e3340f49437403c"
    "8bcfbb62e4252d13de243eb1d606b74738a5ba353da3af85369bbec3aa91b6bd"
    "c618bea85db936b672fdbd50fd643f6ef73f40e99e1ebce89109b997b5f33fb2"
    "f0093d3ada993f2e4ea0b47632b03a695048393a109d3effff3fc063f5f03f29"
    "61803d81a7233fe9a7b33bcd7e283cddd23a4078e5eeb355e2babdc3ff3fc072"
    "da38c058717e3da17940bee2e66abd48882840cce3e5bb06b082afa13f73bf63"
    "bdc4bb95b4653c1f8568bf1224203c8dcef7be94d34632fcc703b0866f0db139"
    "e83fc0f8d33a4036d7b63fe2efcebf940affbe34f1ab36b4813f408086d6b54c"
    "b8943f23798ebe398936c0042c23c068e92740fcc62cc0b30f863f6b8f023ef7"
    "bb78318e40eeb96a252bc0c5b87ebcdc30773e37acc93751e23dc07b981676df"
    "4d42406a2cbe92d530343f"
)


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config files on this branch reference paths relative to the repository root."""
    monkeypatch.chdir(REPO_ROOT)


def _capture_after_deterministic_steps() -> tuple[int, bytes]:
    """Initialize the golden module, step it, and run the ngen save sequence."""
    model = bmi_lstm.bmi_LSTM()
    model.initialize(str(GOLDEN_CONFIG))
    assert set(model.get_input_var_names()) == set(INPUT_MULTIPLIERS)
    for step in range(1, CAPTURE_STEPS + 1):
        for name, multiplier in INPUT_MULTIPLIERS.items():
            model.set_value(name, np.array([step * multiplier * 0.5], dtype="float64"))
        model.update()
    model.set_value(bmi_lstm.SERIALIZATION_CREATE, TRIGGER)
    size = int(model.get_value_ptr(bmi_lstm.SERIALIZATION_SIZE)[0])
    payload = model.get_value_ptr(bmi_lstm.SERIALIZATION_STATE).tobytes()
    model.set_value(bmi_lstm.SERIALIZATION_FREE, TRIGGER)
    return size, payload


def test_golden_module_capture_matches_golden_payload():
    size, payload = _capture_after_deterministic_steps()
    assert size == len(payload) == len(REAL_MODULE_PAYLOAD)

    golden = codec.unpack(REAL_MODULE_PAYLOAD)
    actual = codec.unpack(payload)

    # header, fingerprint, and the member header are exact on every platform
    first_float = codec.HEADER_SIZE + len(golden.fingerprint) + codec.MEMBER_HEADER_SIZE
    assert payload[:first_float] == REAL_MODULE_PAYLOAD[:first_float]

    # the state and output sections are torch results, which differ in the
    # last bits between CPU architectures and torch builds
    assert len(actual.members) == len(golden.members)
    for (hidden, cell), (golden_hidden, golden_cell) in zip(actual.members, golden.members):
        np.testing.assert_allclose(hidden, golden_hidden, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cell, golden_cell, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual.outputs, golden.outputs, rtol=1e-5, atol=1e-9)


def test_real_golden_bytes_describe_the_golden_module():
    """The literal's header matches the module it was captured from."""
    snapshot = codec.unpack(REAL_MODULE_PAYLOAD)
    assert snapshot.timestep == CAPTURE_STEPS
    assert snapshot.fingerprint == (
        b"lstm-bmi;members=1;0:hidden=126,"
        b"inputs=APCP_surface|TMP_2maboveground|elev_mean|slope_mean,"
        b"run=nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806,"
        b"epoch=9"
    )
    assert [hidden.size for hidden, _ in snapshot.members] == [126]
    assert snapshot.outputs.size == 2
