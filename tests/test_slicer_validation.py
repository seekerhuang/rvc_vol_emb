import numpy as np
import pytest

from infer.lib.slicer2 import Slicer, get_rms


@pytest.mark.parametrize("sample_rate", [0, -1, -16000])
def test_slicer_rejects_non_positive_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        Slicer(sr=sample_rate)


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_slicer_preserves_short_audio_at_valid_sample_rate(sample_rate):
    audio = np.zeros(100, dtype=np.float32)
    chunks = Slicer(sr=sample_rate).slice(audio)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], audio)


@pytest.mark.parametrize("hop_size", [0, -1, -20])
def test_slicer_rejects_non_positive_hop_size(hop_size):
    with pytest.raises(ValueError, match="hop_size must be positive"):
        Slicer(sr=16000, hop_size=hop_size)


@pytest.mark.parametrize("hop_size", [1, 20])
def test_slicer_accepts_positive_hop_size(hop_size):
    assert Slicer(sr=16000, hop_size=hop_size).hop_size > 0


@pytest.mark.parametrize("sample_rate,hop_size", [(1, 20), (16000, 0.001)])
def test_slicer_rejects_hop_size_that_rounds_to_zero_samples(sample_rate, hop_size):
    with pytest.raises(ValueError, match="hop_size must span at least one sample"):
        Slicer(sr=sample_rate, hop_size=hop_size)


def test_slicer_keeps_default_hop_size_in_samples():
    assert Slicer(sr=16000).hop_size == 320


@pytest.mark.parametrize("frame_length,hop_length", [(0, 1), (-1, 1), (8, 0), (8, -1)])
def test_rms_rejects_non_positive_window_or_hop(frame_length, hop_length):
    with pytest.raises(ValueError, match="must be positive"):
        get_rms(np.ones(32), frame_length=frame_length, hop_length=hop_length)


def test_rms_of_constant_signal_is_one_away_from_padding():
    rms = get_rms(np.ones(32), frame_length=8, hop_length=2)
    assert rms.shape == (1, 17)
    np.testing.assert_allclose(rms[0, 2:-2], 1.0)
