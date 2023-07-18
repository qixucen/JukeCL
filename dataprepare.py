import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader

# 读取音乐，默认读取7.99s的片段，采样率为16000
def load_wav(music_path, sr=16000, return_torch=False):  
    duration = 7.99
    y, _ = librosa.load(music_path, sr=sr, duration=None)
    num_slices = int(len(y) / (sr * duration))
    slices = []
    for i in range(num_slices):
        start = int(i * sr * duration)
        end = int((i + 1) * sr * duration)
        slice = y[start:end]
        slices.append(slice)

    if return_torch:
        return [torch.Tensor(slice)[np.newaxis, :] for slice in slices]
    # [1, 127840]
    return [np.array(slice)[np.newaxis, :] for slice in slices]


# 对读取的音乐切片提取log-amplitude mel spectrograms特征，n_mels和hop_length均参考原论文
def log_mel(wav, sr=16000):
    if len(wav.shape) > 1:
        wav = wav.reshape(-1)
    if isinstance(wav, torch.Tensor):
        wav = np.array(wav)
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=96, hop_length=160)
    log_S = np.log(S + 1e-8)[np.newaxis, :]  # [1, 96, 800]
    return torch.FloatTensor(log_S)


'''class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x'''

# GTZAN的dataset定义
class GTZAN(Dataset):

    def __init__(self, root, transform=None):
        self.dataset = []
        self.transform = transform
        contents = os.listdir(root)
        folders = [
            folder for folder in contents
            if os.path.isdir(os.path.join(root, folder))
        ]
        self.label2index = dict()
        print('preparing the dataset')
        taqadum = tqdm(enumerate(folders))
        for i, f in taqadum:
            self.label2index[f] = i
            taqadum.set_description('loading {} music'.format(f))
            for r, _, files in os.walk('{}/{}'.format(root, f)):
                for file in files:
                    wav_path = os.path.join(r, file)
                    # if wav_path[-3:] != 'wav':
                    #     break
                    raw_wav = load_wav(wav_path, return_torch=True)
                    if self.transform is not None: 
                        if isinstance(raw_wav, list):
                            for w in raw_wav:
                                self.dataset.append(w)
                        else:
                            self.dataset.append(raw_wav)
                    else:
                        if isinstance(raw_wav, list):
                            for w in raw_wav:
                                self.dataset.append((log_mel(w), torch.Tensor([i])))
                        else:
                            self.dataset.append(
                                (log_mel(load_wav(wav_path)), torch.Tensor([i])))
                    # break

    def __getitem__(self, idx):
        if self.transform is not None:
            wav = self.transform(self.dataset[idx])  # 每次读取数据时都会经过一次增强
            wav_i = log_mel(wav[0])
            wav_j = log_mel(wav[1])
            return wav_i, wav_j
        audio = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return audio, label

    def __len__(self):
        return len(self.dataset)


# 对于支持的dataset设定一个统一的读取函数
# 至于如何定义新的dataset，参考data/README.md中对于数据文件结构的说明
def get_dataset(dataset_dir, dataset='GTZAN', transform=None):
    assert os.path.exists(dataset_dir)
    if dataset == 'GTZAN':
        d = GTZAN(root=dataset_dir, transform=transform)
    return d


# if __name__ == '__main__':
#     dataset = GTZAN(r'./data/GZTAN')
#     for d in dataset:
#         print(d[0].shape, d[1].shape)
#         print(d[0].requires_grad)
#         break
#     # dataset = ClusteringDataset(dataset)
#     print(len(dataset))
#     dataloader = DataLoader(
#         dataset,
#         batch_size=64,
#         num_workers=16,
#         persistent_workers=True,
#         drop_last=False,
#         shuffle=True,
#     )
#     for batch in dataloader:
#         print(batch[0].requires_grad)
#         print(batch[0].shape, batch[1].shape)
#         break
