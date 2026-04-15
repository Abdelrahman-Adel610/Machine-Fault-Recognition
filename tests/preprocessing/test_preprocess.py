import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import soundfile as sf
import pytest
from src.preprocessing.preprocess import preprocess_audio

def test_preprocess_11s_with_silence(tmp_path):
    """
    Tests an 11-second audio file where the last 2 seconds are completely silent.
    Proves that Silence Trimming, Normalization, and Windowing all work perfectly.
    """
    
    # 1. CREATE SYNTHETIC TEST DATA
    target_sr = 16000
    
    # Generate 9 seconds of "fake machine noise" (a simple sine wave)
    t = np.linspace(0, 9, 9 * target_sr)

    # We multiply by 0.5 so the raw max amplitude is 0.5 (to test normalization later)
    noise = 0.5 * np.sin(2 * np.pi * 440 * t) 
    
    # Generate 2 seconds of pure silence
    silence = np.zeros(2 * target_sr)
    
    # Combine them to make a full 11-second file
    full_audio = np.concatenate((noise, silence))
    
    # Save it to a temporary directory provided by pytest (tmp_path)
    test_file = tmp_path / "test_11s_silence.wav"
    sf.write(test_file, full_audio, target_sr)
    
    # 2. RUN THE PIPELINE
    result = preprocess_audio(
        file_path=str(test_file), 
        target_sr=target_sr, 
        chunk_duration=3.0, 
        step_duration=1.5
    )
    
    
    # Check A: Output format is correct
    assert isinstance(result, np.ndarray), "Output must be a Numpy array"
    
    # Check B: Fixed Length padding/windowing is perfect
    # 3 seconds * 16000 Hz = 48000 samples per chunk
    assert result.shape[1] == 48000, f"Expected 48000 samples per chunk, got {result.shape[1]}"
    
    # Check C: Volume Normalization worked
    # The raw audio max was 0.5, but the pipeline should force it to exactly 1.0
    assert np.isclose(np.max(np.abs(result)), 1.0), "Volume was not normalized to 1.0"
    
    # Check D: Silence Trimming worked
    # An 11s file normally creates 7 chunks. 
    # But because our pipeline trims the 2s of silence, it acts like a 9s file.
    # A 9s file sliced in 3s chunks with a 1.5s step creates exactly 5 or 6 chunks.
    assert result.shape[0] in [5, 6], f"Silence trimming failed! Expected 5 or 6 chunks, got {result.shape[0]}"


def test_preprocess_short_audio(tmp_path):
    """
    Tests an audio file that is only 2 seconds long (shorter than the 3s chunk).
    Verifies that the pipeline correctly pads the audio with zeros to 3s.
    """
    target_sr = 16000
    
    # 1. Create a 2-second dummy signal
    t = np.linspace(0, 2, 2 * target_sr)
    short_noise = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    test_file = tmp_path / "test_2s.wav"
    sf.write(test_file, short_noise, target_sr)
    
    # 2. Run the pipeline
    result = preprocess_audio(
        file_path=str(test_file), 
        target_sr=target_sr, 
        chunk_duration=3.0
    )
    
    # 3. Assertions
    # With 2s audio and 3s target, we should get EXACTLY 1 chunk
    assert result.shape[0] == 1, "Short audio should produce exactly 1 chunk"
    
    # The chunk length must be 3 seconds (48,000 samples)
    assert result.shape[1] == 48000, "Short audio was not padded to 3 seconds"
    
    # Check that it actually padded with zeros (the last 1 second should be 0)
    # The original noise was only 32,000 samples long (2s * 16k)
    # So indices from 32,000 to 47,999 should be 0.0
    assert np.all(result[0, 32000:] == 0.0), "Padding was not applied correctly (expected zeros)"