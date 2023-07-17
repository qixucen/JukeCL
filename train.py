import os
import argparse
import torch
import torch.nn as nn
from model.music import MCB, GRU, DEMCB
from module.learning import ClusteringLearning
from module.utils import yaml_config_hook
from dataprepare import get_dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

device = torch.device('cuda:0')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
'''from torchaudio_augmentations import (
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
)'''


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


if __name__ == '__main__':
    MCBdims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50),
            (128, 1, 25)]
    GRUdims = [(128, 25), (256, 25), (128, 25)]
    autoencoder = MusicAutoEncoder(MCBdims=MCBdims,
                                GRUdims=GRUdims,
                                feature_dim=32)
    input = torch.randn(12, 1, 96, 800)
    # print(autoencoder(input)[0].shape, autoencoder(input)[1].shape)
    # exit(0)

    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # train_transform = [
    #     RandomResizedCrop(n_samples=args.audio_length),
    #     RandomApply([PolarityInversion()], p=args.transforms_polarity),
    #     RandomApply([Noise()], p=args.transforms_noise),
    #     RandomApply([Gain()], p=args.transforms_gain),
    #     RandomApply([HighLowPass(sample_rate=args.sample_rate)],
    #                 p=args.transforms_filters),
    #     RandomApply([Delay(sample_rate=args.sample_rate)],
    #                 p=args.transforms_delay),
    #     RandomApply(
    #         [
    #             PitchShift(
    #                 n_samples=args.audio_length,
    #                 sample_rate=args.sample_rate,
    #             )
    #         ],
    #         p=args.transforms_pitch,
    #     ),
    #     RandomApply([Reverb(sample_rate=args.sample_rate)],
    #                 p=args.transforms_reverb),
    # ]

    dataset = get_dataset(args.dataset_dir, args.dataset)

    # dataset = ClusteringDataset(dataset=dataset)
    # train_dataset = ContrastiveDataset(
    #     train_dataset,
    #     input_shape=(1, args.audio_length),
    #     transform=ComposeMany(
    #         train_transform, num_augmented_samples=2
    #     ),
    # )

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=16,
    #     persistent_workers=True,
    #     drop_last=True,
    #     shuffle=True,
    # )

    # print("!")
    # for d in train_loader:
    #     print(d[0].shape)

    module = ClusteringLearning(args, autoencoder).to(device)
    batch_x = []
    batch_y = []
    batch_size = args.batch_size
    count = 0
    for d in dataset:
        count += 1
        batch_x.append(d[0].unsqueeze(0))
        if count == batch_size:
            batch_x = torch.cat(batch_x, dim=0).to(device)
            print(module.training_step(batch_x))
            batch_x = []
            batch_y = []
            count = 0
        else:
            continue
# logger = TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset))
# if args.checkpoint_path:
#     pass
# else:
#     trainer = Trainer(
#         logger=logger,
#         sync_batchnorm=True,
#         max_epochs=args.max_epochs,
#         log_every_n_steps=20,
#         check_val_every_n_epoch=1,
#         accelerator=args.accelerator,
#         # gpus=[0],
#         devices=1,
#     )
#     trainer.fit(module, train_dataloaders=train_loader)

# print(autoencoder.parameters)
