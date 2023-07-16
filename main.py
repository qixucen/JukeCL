import argparse
import torch.nn as nn
from model.music import MCB, GRU
from module.learning import ContrastiveLearning
from module.utils import yaml_config_hook
from dataset import get_dataset, ContrastiveDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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

class MusicEncoder(nn.Module):

    def __init__(self, MCBdims, GRUdims, feature_dim=32) -> None:
        super().__init__()
        self.MCB = MCB(dims=MCBdims)
        self.GRU = GRU(dims=GRUdims)
        self.fc = nn.Linear(GRUdims[-1][0] * GRUdims[-1][1], feature_dim)

    def forward(self, input):
        output = self.MCB(input)
        output = self.GRU(output)
        output = output.flatten(-2)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    MCBdims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50),
               (128, 1, 25)]
    GRUdims = [(128, 25), (256, 25), (128, 25)]
    encoder = MusicEncoder(MCBdims=MCBdims, GRUdims=GRUdims, feature_dim=32)

    parser = argparse.ArgumentParser(description="CLMR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    
    
    train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
            ),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=args.audio_length,
                        sample_rate=args.sample_rate,
                    )
                ],
                p=args.transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            ),]
    
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train", download=False)

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=2
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
    )
  
    module = ContrastiveLearning(args, encoder)
    logger = TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset))
    if args.checkpoint_path:
        module = module.load_from_checkpoint(
            args.checkpoint_path, encoder=encoder, output_dim=train_dataset.n_classes
        )
    else:
        trainer = Trainer(
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
            gpus=[0],
        )
        trainer.fit(module, train_loader, valid_loader)
    
    print(encoder.parameters)
    print(encoder.fc.in_features)
    # cl = ContrastiveLearning(args=args, encoder=encoder)
