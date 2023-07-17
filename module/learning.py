from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from simclr import SimCLR
from simclr.modules import NT_Xent, LARS


class ContrastiveLearning(LightningModule):

    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.n_features = (self.encoder.fc.in_features
                           )  # get dimensions of last fully-connected layer
        self.model = SimCLR(  # in SimCLR, the encofer.fc will be replaced with nn.Indentity()
            self.encoder, self.hparams.projection_dim, self.n_features)
        # self.model = Dino(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = self.configure_criterion()

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        _, _, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, _) -> torch.Tensor:
        x, _ = batch
        x_i = x[:, 0, :]
        x_j = x[:, 1, :]
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        # PT lightning aggregates differently in DP mode
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=1)
        return criterion

    def configure_optimizers(self) -> dict:
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * self.hparams.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.hparams.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=0, last_epoch=-1)
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


class ClusteringLayer(nn.Module):

    def __init__(self,
                 n_clusters=10,
                 hidden_size=10,
                 cluster_centers=None,
                 alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden_size = hidden_size
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters,
                                                  self.hidden_size,
                                                  dtype=torch.float).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = torch.nn.Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (
            numerator.t() /
            torch.sum(numerator, 1)).t()  #soft assignment using t-distribution
        return t_dist


class ClusteringLearning(nn.Module):

    def __init__(self, args, autoencoder: nn.Module) -> None:
        super().__init__()
        self.args = args
        self.autoencoder = autoencoder
        self.reconLoss = nn.MSELoss()
        self.Kldivgence = nn.KLDivLoss(reduction='mean')

        self.num_clusters = args.num_clusters
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.cluster_centers = None
        self.clusteringlayer = ClusteringLayer(self.num_clusters,
                                               self.hidden_size,
                                               self.cluster_centers,
                                               self.alpha)
        self.optimizer = self.configure_optimizers()

    def target_distribution(self, q_):
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, input):
        hidden, recon = self.autoencoder.forward(input)
        return hidden, recon

    def training_step(self, input):
        hidden, recon = self(input)
        dec = self.clusteringlayer(hidden)
        target = self.target_distribution(dec).detach()
        loss = self.args.alpha * self.reconLoss(
            input, recon) + (1 - self.args.alpha) * self.Kldivgence(
                dec.log(), target) / dec.shape[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(list(self.autoencoder.parameters()) +
                                     list(self.clusteringlayer.parameters()),
                                     lr=self.args.learning_rate)
        # return {"optimizer": optimizer}
        return optimizer