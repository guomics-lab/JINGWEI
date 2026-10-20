import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from ..ae_models import build_dcae


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
        supervision_mask_tensor=None,
        num_workers=0,
        ae_architecture="global",
        ae_dropout=.1,
    ):
        super().__init__()
        supervision = (
            full_mask_tensor
            if supervision_mask_tensor is None
            else supervision_mask_tensor
        )
        self.register_buffer(
            "full_mask_tensor", full_mask_tensor.clone(), persistent=False
        )
        self.register_buffer(
            "supervision_mask_tensor", supervision.clone(), persistent=False
        )
        if (
            full_data_tensor.shape != full_mask_tensor.shape
            or full_mask_tensor.shape != self.supervision_mask_tensor.shape
        ):
            raise ValueError(
                "data, input mask and supervision mask must have matching shapes"
            )
        for mask in (self.full_mask_tensor, self.supervision_mask_tensor):
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("masks must contain only 0 and 1")
        if torch.any(self.full_mask_tensor > self.supervision_mask_tensor):
            raise ValueError(
                "training input must be a subset of supervised observations"
            )
        self.register_buffer(
            "full_data_tensor",
            torch.where(self.supervision_mask_tensor.bool(), full_data_tensor, 0.0),
            persistent=False,
        )
        if not torch.isfinite(self.full_data_tensor).all():
            raise ValueError("supervised training values must be finite")

        self.save_hyperparameters(
            ignore=["full_data_tensor", "full_mask_tensor", "supervision_mask_tensor"]
        )

        feature_dim = self.full_data_tensor.size(1)

        self.ae_architecture = ae_architecture
        self.model = build_dcae(ae_architecture, dropout=ae_dropout,
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
        self.num_workers = num_workers

    def forward(self, x, observed_mask=None):
        return self.model(x, observed_mask) if getattr(self.model, "mask_aware", False) else self.model(x)

    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(
            self.full_data_tensor * self.full_mask_tensor,
            self.full_data_tensor,
            self.full_mask_tensor,
            self.supervision_mask_tensor,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def training_step(self, batch, batch_idx):
        input_batch, target_batch, mask_batch, supervision_batch = batch
        observed = supervision_batch > 0.5
        x_recon, mask_logits = self(input_batch, mask_batch)

        if observed.any():
            recon_loss = F.mse_loss(x_recon[observed], target_batch[observed])
        else:
            recon_loss = x_recon.sum() * 0.0

        if self.lambda_mask > 0:
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits, mask_batch.float()
            )
        else:
            mask_loss = mask_logits.sum() * 0.0

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
            input_data = self.full_data_tensor * self.supervision_mask_tensor
            x_tilde, _ = self(input_data, self.supervision_mask_tensor)
            imputed_data = self.full_data_tensor.clone()

            unobserved_mask = self.supervision_mask_tensor == 0
            imputed_data[unobserved_mask] = x_tilde[unobserved_mask]

            return imputed_data
