import librosa
import numpy as np
import warnings

# Ignore librosa warnings about reading standard wav files to keep console clean
warnings.filterwarnings('ignore')

def preprocess_audio(file_path, target_sr, chunk_duration, step_duration, trim_top_db=20):
    """
    Reads a raw audio file, cleans it, and splits it into fixed-size overlapping chunks.
    
    Parameters:
    - file_path: Path to the .wav file.
    - target_sr: Target sample rate (16000Hz is standard for DL and fast).
    - chunk_duration: Length of each slice in seconds (e.g., 3.0).
    - step_duration: How far to slide the window in seconds (1.5s means 50% overlap).
    
    Returns:
    - np.array of shape (num_chunks, chunk_length)
    """
    
    # STEP 1: Load & Resample (Handles Microphone Variations)
    y, sr = librosa.load(file_path, sr=target_sr, res_type='soxr_hq')
    # STEP 2: Silence Trimming (Handles Dead Space)

    y_trimmed, _ = librosa.effects.trim(y, top_db=trim_top_db)

    
    # STEP 3: Peak Normalization (Handles Volume Variations)
    max_amplitude = np.max(np.abs(y_trimmed))
    if max_amplitude > 0:
        y_normalized = y_trimmed / max_amplitude
    else:
        y_normalized = y_trimmed 
        
    # STEP 4: Windowing / Slicing (Solution 2 - Prevents Data Loss)
    chunk_length = int(target_sr * chunk_duration) 
    step_length = int(target_sr * step_duration)
    
    chunks = []
    
    # CASE A: The audio is shorter than our target chunk (e.g., 2 seconds)
    if len(y_normalized) < chunk_length:
        # Pad with zeros at the end to force it to be exactly 3 seconds
        pad_length = chunk_length - len(y_normalized)
        y_padded = np.pad(y_normalized, (0, pad_length), mode='constant')
        chunks.append(y_padded)
        
    # CASE B: The audio is longer (e.g., 7 seconds) -> Slice it!
    else:
        start = 0
        while start + chunk_length <= len(y_normalized):
            chunks.append(y_normalized[start : start + chunk_length])
            start += step_length
            
        # Catch the "tail" of the audio if it doesn't divide perfectly
        # We take the LAST 3 seconds of the file to ensure we don't miss the ending
        if start < len(y_normalized):
            chunks.append(y_normalized[-chunk_length:])
            
    # Return as a 2D numpy array: shape = (number_of_chunks, samples_per_chunk)
    return np.array(chunks)