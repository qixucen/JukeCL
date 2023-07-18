import os
import argparse
import torch
import numpy as np
from module.learning import ClusteringLearning, ContrastiveLearning
from module.utils import yaml_config_hook
from dataprepare import GTZAN
from torch.utils.data import DataLoader
from train import MusicAutoEncoder, MusicEncoder
from visualization import T_SNE

device = torch.device('cuda:0')

MCBdims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50),
           (128, 1, 25)]
GRUdims = [(128, 25), (256, 25), (128, 25)]
autoencoder = MusicEncoder(MCBdims=MCBdims,
                           GRUdims=GRUdims,
                           feature_dim=32)

parser = argparse.ArgumentParser(description="CLMR")
config = yaml_config_hook("./config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
args = parser.parse_args()


# module = ContrastiveLearning(args, autoencoder).to(device)
# checkpoint_path = '/home/ubuntu/AI/clmr/Unsupervised-Rhythm-Clustering-Embedding/runs/CLMRv2-GTZAN-contrastive/version_0/checkpoints/epoch=35-step=3564.ckpt'
# module = module.load_from_checkpoint(
#     checkpoint_path=checkpoint_path, encoder=autoencoder, args=args)

encoder = MusicEncoder(MCBdims=MCBdims,
                               GRUdims=GRUdims,
                               feature_dim=32)
module = ContrastiveLearning(args, encoder).to(device)
checkpoint_path = args.upstream_checkpoint_path
module = module.load_from_checkpoint(checkpoint_path=checkpoint_path, encoder=encoder, args=args)

checkpoint_path = args.downstream_checkpoint_path
module = ClusteringLearning(args, module.encoder).to(device)
module = module.load_from_checkpoint(checkpoint_path=checkpoint_path, encoder=module.encoder, args=args)

dataset = GTZAN(os.getcwd() + args.dataset_dir)
label2index = dataset.label2index
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    shuffle=True,
)
embeddings = []
labels = []
for batch in train_loader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    embeddings.append(module.encoder(x).cpu().detach().numpy())
    labels.append(y.cpu().detach().numpy())
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)
print(embeddings.shape)
print(labels.shape)
T_SNE(embeddings, labels)
