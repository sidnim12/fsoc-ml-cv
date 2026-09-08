from __future__ import annotations

import pytest

from src.temporal_verifier import TemporalConfig, TemporalVerifier, temporal_config_from_mapping


def test_persistence_temporal_verification() -> None:
    verifier = TemporalVerifier(TemporalConfig(history_window=5, min_confirmations=3, association_radius_px=25))

    first = verifier.update(0, {"candidate_id": 1, "x_px": 100.0, "y_px": 50.0})
    second = verifier.update(1, {"candidate_id": 1, "x_px": 106.0, "y_px": 52.0})
    third = verifier.update(2, {"candidate_id": 1, "x_px": 112.0, "y_px": 54.0})

    assert first["temporally_verified"] is False
    assert second["confirmation_count"] == 2
    assert third["temporally_verified"] is True
    assert third["confirmation_count"] == 3
    assert third["verification_mode"] == "persistence"
    assert third["blink_match"] is None


def test_persistence_breaks_on_missing_frame() -> None:
    verifier = TemporalVerifier(TemporalConfig(history_window=5, min_confirmations=3))

    verifier.update(0, {"x_px": 100.0, "y_px": 100.0})
    verifier.update(1, {"x_px": 105.0, "y_px": 100.0})
    missing = verifier.update(2, None)

    assert missing["temporally_verified"] is False
    assert missing["confirmation_count"] == 0


def test_persistence_rejects_jump_away_from_expected_position() -> None:
    verifier = TemporalVerifier(TemporalConfig(history_window=5, min_confirmations=2, association_radius_px=10))

    verifier.update(0, {"x_px": 50.0, "y_px": 50.0})
    result = verifier.update(1, {"x_px": 90.0, "y_px": 90.0}, expected_position=(55.0, 50.0))

    assert result["temporally_verified"] is False
    assert result["confirmation_count"] == 0


def test_optional_blink_verification_logic() -> None:
    verifier = TemporalVerifier(
        TemporalConfig(
            mode="blink",
            history_window=10,
            min_confirmations=2,
            expected_period_frames=6,
            tolerance_frames=1,
        )
    )

    verifier.update(0, {"x_px": 10.0, "y_px": 10.0})
    verifier.update(1, None)
    verifier.update(2, None)
    verifier.update(3, None)
    verifier.update(4, None)
    verifier.update(5, None)
    result = verifier.update(6, {"x_px": 11.0, "y_px": 10.0})

    assert result["verification_mode"] == "blink"
    assert result["blink_match"] is True
    assert result["temporally_verified"] is True


def test_blink_rejects_wrong_period() -> None:
    verifier = TemporalVerifier(
        TemporalConfig(
            mode="blink",
            history_window=10,
            min_confirmations=2,
            expected_period_frames=6,
            tolerance_frames=1,
        )
    )

    verifier.update(0, {"x_px": 10.0, "y_px": 10.0})
    result = verifier.update(3, {"x_px": 11.0, "y_px": 10.0})

    assert result["blink_match"] is False
    assert result["temporally_verified"] is False


def test_temporal_config_from_mapping_reuses_old_keys() -> None:
    config = temporal_config_from_mapping({"enabled": True, "max_history": 7, "min_history": 4})

    assert config.history_window == 7
    assert config.min_confirmations == 4


def test_invalid_temporal_configuration() -> None:
    with pytest.raises(ValueError, match="mode"):
        TemporalVerifier(TemporalConfig(mode="unknown"))
    with pytest.raises(ValueError, match="history_window"):
        TemporalVerifier(TemporalConfig(history_window=0))
    with pytest.raises(ValueError, match="min_confirmations"):
        TemporalVerifier(TemporalConfig(history_window=2, min_confirmations=3))
