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
