import argparse
import torch.nn as nn
from model.music import MCB, GRU
from module.learning import ContrastiveLearning
from module.utils import yaml_config_hook


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
    
    print(encoder.parameters)
    print(encoder.fc.in_features)
    # cl = ContrastiveLearning(args=args, encoder=encoder)
