import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from .models import DCAE


class DCAEImputer(pl.LightningModule):
    def __init__(
        self,
        full_data_tensor,
        full_mask_tensor,
        ae_dim=64,
        mask_predictor_hidden_dim=128,
        lambda_mask=0,
        num_encoder_blocks=2,
        num_decoder_blocks=2,
        dilation=2,
        learning_rate=2e-5,
        batch_size=64,
    ):
        super().__init__()
        self.full_data_tensor = full_data_tensor
        self.full_mask_tensor = full_mask_tensor

        self.save_hyperparameters(ignore=["full_data_tensor", "full_mask_tensor"])

        feature_dim = self.full_data_tensor.size(1)

        self.model = DCAE(
            input_feature_dim=feature_dim,
            ae_dim=ae_dim,
            mask_predictor_hidden_dim=mask_predictor_hidden_dim,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            dilation=dilation,
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_mask = lambda_mask

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
        observed = mask_batch > 0.5
        x_recon, mask_logits = self.model(input_batch)

        if observed.any():
            recon_loss = F.mse_loss(x_recon[observed], target_batch[observed])
        else:
            recon_loss = torch.zeros([], device=self.device)

        if self.lambda_mask > 0:
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_batch.float())
        else:
            mask_loss = torch.zeros([], device=self.device)

        loss = recon_loss + self.lambda_mask * mask_loss
        self.log_dict(
            {"train/loss": loss, "train/recon": recon_loss, "train/mask": mask_loss},
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        return optimizer

    def get_imputed_data(self):
        self.eval()
        with torch.no_grad():
            input_data = self.full_data_tensor * self.full_mask_tensor
            x_tilde, _ = self.model(input_data)
            imputed_data = self.full_data_tensor.clone()

            unobserved_mask = self.full_mask_tensor == 0
            imputed_data[unobserved_mask] = x_tilde[unobserved_mask]

            return imputed_data
