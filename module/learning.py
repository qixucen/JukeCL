from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from simclr import SimCLR
from simclr.modules import NT_Xent, LARS


'''class ContrastiveLearning(LightningModule):

    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.encoder = encoder
        self.n_features = args.hidden_size
        self.model = SimCLR(  # in SimCLR, the encofer.fc will be replaced with nn.Indentity()
            self.encoder, self.hparams.projection_dim, self.n_features)
        
        self.contrastive_criterion = self.configure_criterion()

        self.Kldivgence = nn.KLDivLoss(reduction='sum')
        self.num_clusters = args.num_clusters
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.cluster_centers = None
        self.clusteringlayer = ClusteringLayer(self.num_clusters,
                                               self.hidden_size,
                                               self.cluster_centers,
                                               self.alpha)
        
    def target_distribution(self, q_):
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        # print(z_i.shape, z_j.shape)
        conloss = self.contrastive_criterion(z_i, z_j)

        group = torch.cat([h_i, h_j], dim=0)
        dec = self.clusteringlayer(group)
        target = self.target_distribution(dec).detach()
        k = self.args.k * 100
        conloss *= k
        klloss = (1 - k) * self.Kldivgence(
            dec.log(), target) 
        print(conloss.item(), klloss.item())
        return conloss + klloss

    def training_step(self, batch) -> torch.Tensor:
        x_i, x_j = batch
        # x_i = x[:, 0, :]
        # x_j = x[:, 1, :]
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
'''


class ContrastiveLearning(LightningModule):

    def __init__(self, args, encoder: nn.Module, hidden_size = 32):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.encoder = encoder
        self.n_features = hidden_size
        self.model = SimCLR(  # in SimCLR, the encofer.fc will be replaced with nn.Indentity()
            self.encoder, self.hparams.projection_dim, self.n_features)

        self.contrastive_criterion = self.configure_criterion()

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        _, _, z_i, z_j = self.model(x_i, x_j)
        # print(z_i.shape, z_j.shape)
        conloss = self.contrastive_criterion(z_i, z_j)
        return conloss

    def training_step(self, batch) -> torch.Tensor: # batch: x_i, x_j
        x_i, x_j = batch
        # x_i = x[:, 0, :]
        # x_j = x[:, 1, :]
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
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
                 hidden_size=32,
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
            nn.init.xavier_uniform_(initial_cluster_centers)  # [10, 32]
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = torch.nn.Parameter(initial_cluster_centers)

    def forward(self, x):  # x:[B, 32]
        norm_squared = torch.sum(
            (x.unsqueeze(1) - self.cluster_centers)**2, 2)  # [B, 10]
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (
            numerator.t() /
            torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class ClusteringLearning(LightningModule):

    def __init__(self, args, encoder: nn.Module, cluster_centors=None) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.encoder = encoder
        self.reconLoss = nn.MSELoss()
        self.Kldivgence = nn.KLDivLoss(reduction='sum')

        self.num_clusters = args.num_clusters
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.cluster_centers = cluster_centors
        self.clusteringlayer = ClusteringLayer(self.num_clusters,
                                               self.hidden_size,
                                               self.cluster_centers,
                                               self.alpha)

    def target_distribution(self, q_):
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, input):
        hidden = self.encoder.forward(input)
        return hidden

    def training_step(self, batch):  # batch: x, y
        input = batch[0]
        hidden = self(input)  # hidden: [B, 32]
        dec = self.clusteringlayer(hidden)
        target = self.target_distribution(dec).detach()
        klloss = self.Kldivgence(
            dec.log(), target)
        self.log('Train/loss', klloss)
        return klloss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.clusteringlayer.parameters()),
                                     lr=self.args.learning_rate)
        # return {"optimizer": optimizer}
        return optimizer
