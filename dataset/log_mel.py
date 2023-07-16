import os
import librosa
import torch
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_wav(music_path, sr=16000):
    y, _ = librosa.load(music_path, sr=sr, duration=7.99)
    return np.array(y)[np.newaxis, :]  # [1, 127840]


def log_mel(wav, sr=16000):
    if len(wav.shape) > 1:
        wav = wav.reshape(-1)
    if isinstance(wav, torch.Tensor):
        wav = np.array(wav)
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=96, hop_length=160)
    log_S = np.log(S + 1e-8)[np.newaxis, :]  # [1, 96, 800]
    return torch.FloatTensor(log_S)


# path = r'D:\Assignment\dance\Unsupervised-Rhythm-Clustering-Embedding\data\GZTAN\blues\blues.00000.wav'
# print(log_mel(torch.Tensor(load_wav(path))).shape)