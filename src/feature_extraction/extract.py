import numpy as np
import librosa

def extract_mel_spectrograms(chunks, sr, n_fft, hop_length, n_mels):
    features =[]
    for chunk in chunks:
        mel_spec = librosa.feature.melspectrogram(
            y=np.asarray(chunk, dtype=np.float32),
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=1.0)
        features.append(log_mel)
    return np.array(features, dtype=np.float32)
