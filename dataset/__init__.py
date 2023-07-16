import os
import torch
# from torchaudio_augmentations import Compose
from typing import Tuple, List
from torch.utils.data import Dataset
from dataset.log_mel import log_mel, load_wav

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader


class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class GTZAN(Dataset):

    def __init__(self, root):
        self.dataset = []
        contents = os.listdir(root)
        folders = [
            folder for folder in contents
            if os.path.isdir(os.path.join(root, folder))
        ]
        self.label2index = dict()
        for i, f in enumerate(folders):
            self.label2index[f] = i
            for root, _, files in os.walk('{}/{}'.format(root, f)):
                for file in files:
                    wav_path = os.path.join(root, file)
                    if wav_path[-3:] != 'wav':
                        break
                    self.dataset.append(
                        (torch.Tensor(load_wav(wav_path)), torch.Tensor([i])))

    def __getitem__(self, idx):
        audio = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return audio, label

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset_dir, dataset='GTZAN', subset='train'):
    assert os.path.exists(dataset_dir)
    if dataset == 'GTZAN':
        d = GTZAN(root=dataset_dir, subset=subset)
    return d


class ContrastiveDataset(Dataset):

    def __init__(self, dataset: Dataset, input_shape: List[int],
                 transform: Compose):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape
        self.ignore_idx = []

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self.ignore_idx:
            return self[idx + 1]

        audio, label = self.dataset[idx]

        if audio.shape[1] != self.input_shape[1]:
            self.ignore_idx.append(idx)
            return self[idx + 1]

        if self.transform:
            audio = self.transform(audio)
        return log_mel(audio), label

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == '__main__':
    dataset = GTZAN(r'data\GZTAN')
    for d in dataset:
        print(d[0].shape, d[1].shape)
        print(d[0].requires_grad)
        break
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=16,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
    )
    for batch in dataloader:
        print(batch[0].requires_grad)
        print(batch[0].shape, batch[1].shape)
        break