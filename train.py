from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model.music import MCB, GRU, DEMCB
from module.learning import ClusteringLearning, ContrastiveLearning
from module.utils import yaml_config_hook
from dataprepare import get_dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

device = torch.device('cuda:1')

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MusicAutoEncoder(nn.Module):

    def __init__(self, MCBdims, GRUdims, feature_dim=32) -> None:
        super().__init__()
        self.MCBdims = MCBdims
        self.GRUdims = GRUdims
        self.MCB = MCB(dims=MCBdims)
        self.GRU = GRU(dims=GRUdims)
        self.MCB_de = DEMCB(dims=MCBdims[::-1])
        self.GRU_de = GRU(dims=GRUdims[::-1])
        self.projection_in = nn.Linear(GRUdims[-1][0] * GRUdims[-1][1],
                                       feature_dim)
        self.projection_out = nn.Linear(feature_dim,
                                        GRUdims[-1][0] * GRUdims[-1][1])

    def encoder(self, input):
        hidden = self.MCB(input)
        hidden = self.GRU(hidden)
        hidden = hidden.flatten(-2)
        hidden = self.projection_in(hidden)

        return hidden

    def decoder(self, hidden):
        recon = self.projection_out(hidden)
        recon = recon.reshape(-1, *self.GRUdims[-1])
        recon = self.GRU_de(recon)
        recon = recon.unsqueeze(-2)
        recon = self.MCB_de(recon)
        return recon

    def forward(self, input):
        hidden = self.encoder(input)
        recon = self.decoder(hidden)
        return hidden, recon


class MusicEncoder(nn.Module):

    def __init__(self, MCBdims, GRUdims, feature_dim=32) -> None:
        super().__init__()
        self.MCBdims = MCBdims
        self.GRUdims = GRUdims
        self.MCB = MCB(dims=MCBdims)
        self.GRU = GRU(dims=GRUdims)
        self.projection = nn.Linear(GRUdims[-1][0] * GRUdims[-1][1],
                                    feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, input):
        hidden = self.MCB(input)
        hidden = self.GRU(hidden)
        hidden = hidden.flatten(-2)
        hidden = self.projection(hidden)
        hidden = self.fc(hidden)

        return hidden


if __name__ == '__main__':
    # 参考网易原论文Table 7的输出维度
    MCBdims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50),
               (128, 1, 25)]
    GRUdims = [(128, 25), (256, 25), (128, 25)]

    # load args
    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # transform is necessary in Constrative learning(like Simclr)
    transform = [
        RandomResizedCrop(n_samples=args.audio_length),
        RandomApply([PolarityInversion()], p=args.transforms_polarity),
        RandomApply([Noise()], p=args.transforms_noise),
        RandomApply([Gain()], p=args.transforms_gain),
        RandomApply([HighLowPass(sample_rate=args.sample_rate)],
                    p=args.transforms_filters),
        RandomApply([Delay(sample_rate=args.sample_rate)],
                    p=args.transforms_delay),
        RandomApply(
            [
                PitchShift(
                    n_samples=args.audio_length,
                    sample_rate=args.sample_rate,
                )
            ],
            p=args.transforms_pitch,
        ),
        RandomApply([Reverb(sample_rate=args.sample_rate)],
                    p=args.transforms_reverb),
    ]

    if args.task == 'clustering':  # 进行downstream的clustering训练
        dataset = get_dataset(
            dataset='GTZAN', dataset_dir=os.getcwd() + args.dataset_dir, transform=None)
        drop_last = False
    elif args.task == 'contrastive':
        dataset = get_dataset(
            dataset='GTZAN', dataset_dir=os.getcwd() + args.dataset_dir, transform=ComposeMany(
                transform, num_augmented_samples=2
            ))
        drop_last = True  # 由于Simclr计算损失时需要确定的batch_size, 所以drop_last=True是必须的

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=16,
        persistent_workers=True,
        drop_last=drop_last,
        shuffle=True,
    )

    encoder = MusicEncoder(MCBdims=MCBdims,
                           GRUdims=GRUdims,
                           feature_dim=32)
    module = ContrastiveLearning(args, encoder).to(device)

    logger = TensorBoardLogger(
        "runs", name="CLMRv2-{}-{}".format(args.dataset, args.task))

    if args.upstream_checkpoint_path and args.task == 'clustering':

        module = module.load_from_checkpoint(
            checkpoint_path=args.upstream_checkpoint_path, encoder=encoder, args=args)

        # initial the clustering centriod
        embeddings = []
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            embeddings.append(module.encoder(x).cpu().detach().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init='auto').fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).to(device)
        module = ClusteringLearning(args, module.encoder).to(device)
        trainer = Trainer(
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
            devices=1,
        )
        trainer.fit(module, data_loader)
    elif args.task == 'contrastive':
        trainer = Trainer(
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
            devices=1,
        )
        trainer.fit(module, data_loader)
