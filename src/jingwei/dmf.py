import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from .models import DMF


class DMFImputer(pl.LightningModule):
    def __init__(
        self,
        full_data_tensor,
        full_mask_tensor,
        embedding_dim=64,
        hidden_dims=[256, 128],
        reconstruction_weight=1.0,
        mask_weight=0.5,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=1024,
    ):
        super(DMFImputer, self).__init__()

        self.full_data_tensor = full_data_tensor
        self.full_mask_tensor = full_mask_tensor
        self.save_hyperparameters(ignore=["full_data_tensor", "full_mask_tensor"])

        n_samples, n_proteins = self.full_data_tensor.shape
        self.model = DMF(
            n_rows=n_samples,
            n_cols=n_proteins,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            mask_predictor_hidden_dim=hidden_dims[0] if hidden_dims else 128,
        )

        self.batch_size = batch_size
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.reconstruction_weight = reconstruction_weight
        self.mask_weight = mask_weight

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(
            self.full_data_tensor * self.full_mask_tensor,
            self.full_data_tensor,
            self.full_mask_tensor,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20,
            persistent_workers=True,
        )

    def training_step(self, batch, batch_idx):
        input_batch, target_batch, mask_batch = batch
        x_tilde, predicted_mask_logits = self.model(input_batch)
        observed_points = mask_batch == 1
        if observed_points.sum() > 0:
            recon_loss = F.mse_loss(x_tilde[observed_points], target_batch[observed_points])
        else:
            recon_loss = torch.tensor(0.0, device=self.device)

        if self.mask_weight > 0:
            mask_loss = F.binary_cross_entropy_with_logits(
                predicted_mask_logits, mask_batch
            )
            mask_acc = (
                (torch.sigmoid(predicted_mask_logits) > 0.5) == mask_batch
            ).float().mean()
        else:
            mask_loss = torch.tensor(0.0, device=self.device)
            mask_acc = torch.tensor(0.0, device=self.device)

        total_loss = self.reconstruction_weight * recon_loss + self.mask_weight * mask_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True)

        if self.mask_weight > 0:
            self.log("train/mask_loss", mask_loss, on_step=True, on_epoch=True)
            self.log("train/mask_acc", mask_acc, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return optimizer

    def get_imputed_data(self):
        self.eval()
        with torch.no_grad():
            full_matrix = self.model.get_full_matrix()
            imputed_data = self.full_data_tensor.clone()
            unobserved_mask = self.full_mask_tensor == 0
            imputed_data[unobserved_mask] = full_matrix[unobserved_mask]

            return imputed_data
