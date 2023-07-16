import librosa
import numpy as np


def log_mel(music_path):
    y, sr = librosa.load(music_path, sr=16000, duration=7.99)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, hop_length=160)
    log_S = np.log(S + 1e-8)[np.newaxis, :]
    return log_S

# path = r'D:\Assignment\dance\Unsupervised-Rhythm-Clustering-Embedding\data\GZTAN\blues\blues.00000.wav'
# print(log_mel(path).shape)