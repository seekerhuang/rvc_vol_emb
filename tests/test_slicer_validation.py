import numpy as np
import pytest

from infer.lib.slicer2 import Slicer


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
