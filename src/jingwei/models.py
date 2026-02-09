import torch.nn as nn
import torch.nn.functional as F
import torch


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(
            m.weight.data, mode="fan_out", nonlinearity="leaky_relu", a=0.01
        )
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, ResidualDilatedConvLayer):
        for submodule in m.modules():
            if isinstance(submodule, nn.Conv1d):
                nn.init.kaiming_normal_(
                    submodule.weight.data,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                    a=0.01,
                )
                submodule.weight.data *= 0.5


class MaskPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weights_init)

    def forward(self, x_tilde):
        if x_tilde.dim() != 2:
            raise ValueError(
                f"Input to MaskPredictor should be (B, N), got {x_tilde.shape}"
            )

        predicted_logits = self.network(x_tilde)
        return predicted_logits


class LayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm1d, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x


class UpsampleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest", align_corners=False)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.upsample(x))


class ResidualDilatedConvLayer(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=2):
        super(ResidualDilatedConvLayer, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.ln = LayerNorm1d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.ln(out)
        out = self.activation(out)
        return x + out


class EncoderDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=3, num_layers=3):
        super(EncoderDilatedConvBlock, self).__init__()
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = None

        self.layers = nn.ModuleList(
            [
                ResidualDilatedConvLayer(out_channels, kernel_size, dilation)
                for _ in range(num_layers)
            ]
        )
        self.pool = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        return x


class DecoderDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=3, num_layers=3):
        super(DecoderDilatedConvBlock, self).__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = None

        self.layers = nn.ModuleList(
            [
                ResidualDilatedConvLayer(out_channels, kernel_size, dilation)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.upsample(x)
        if self.proj is not None:
            x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvAE(nn.Module):
    def __init__(self, latent_dim=16, num_encoder_blocks=9, num_decoder_blocks=9, dilation=3):
        super(ConvAE, self).__init__()

        self.conv_up = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            LayerNorm1d(4),
            nn.GELU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            LayerNorm1d(8),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            LayerNorm1d(16),
            nn.GELU(),
        )

        encoder_blocks = []
        for _ in range(num_encoder_blocks):
            encoder_blocks.append(
                EncoderDilatedConvBlock(16, 16, kernel_size=3, dilation=dilation)
            )
        self.encoder_blocks = nn.Sequential(*encoder_blocks)
        self.to_latent = nn.Conv1d(16, latent_dim, kernel_size=1)

        decoder_blocks = []
        for i in range(num_decoder_blocks):
            in_ch = latent_dim if i == 0 else 16
            decoder_blocks.append(
                DecoderDilatedConvBlock(in_ch, 16, kernel_size=3, dilation=dilation)
            )
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        self.conv_down = nn.Conv1d(16, 1, kernel_size=3, padding=1)
        self.apply(weights_init)

    def encode(self, x):
        x = self.conv_up(x)
        x = self.encoder_blocks(x)
        latent = self.to_latent(x)
        return latent

    def decode(self, z):
        x = self.decoder_blocks(z)
        x = self.conv_down(x)
        return x

    def forward(self, x):
        original_length = x.shape[2]
        latent = self.encode(x)
        recon = self.decode(latent)

        if recon.shape[2] > original_length:
            recon = recon[:, :, :original_length]
        elif recon.shape[2] < original_length:
            padding_size = original_length - recon.shape[2]
            recon = F.pad(recon, (0, padding_size))

        return recon

    def loss_function(self, x):
        recon = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        return recon_loss


class DCAE(nn.Module):
    def __init__(
        self,
        input_feature_dim,
        ae_dim,
        mask_predictor_hidden_dim=128,
        num_encoder_blocks=3,
        num_decoder_blocks=3,
        dilation=3,
    ):
        super().__init__()
        self.input_feature_dim = input_feature_dim

        self.ae = ConvAE(
            latent_dim=ae_dim,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            dilation=dilation,
        )

        self.mask_predictor = MaskPredictor(
            input_dim=input_feature_dim,
            hidden_dim=mask_predictor_hidden_dim,
            output_dim=input_feature_dim,
        )
        self.apply(weights_init)

    def forward(self, x):
        if x.dim() == 2:
            x_unsqueezed = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x_unsqueezed = x
        else:
            raise ValueError(
                f"Input to DCAE should be (B, N) or (B, 1, N), got {x.shape}"
            )

        recon = self.ae(x_unsqueezed)
        x_tilde = recon.squeeze(1)
        predicted_mask_logits = self.mask_predictor(x_tilde)
        return x_tilde, predicted_mask_logits


class DMF(nn.Module):
    def __init__(
        self, n_rows, n_cols, embedding_dim=64, hidden_dims=[256, 128], mask_predictor_hidden_dim=128
    ):
        super(DMF, self).__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.embedding_dim = embedding_dim
        self.row_embedding = nn.Embedding(n_rows, embedding_dim)
        self.col_embedding = nn.Embedding(n_cols, embedding_dim)

        reconstruction_layers = []
        current_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            reconstruction_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim
        reconstruction_layers.append(nn.Linear(current_dim, 1))
        self.reconstruction_mlp = nn.Sequential(*reconstruction_layers)

        self.last_hidden_dim = hidden_dims[-1] if hidden_dims else embedding_dim * 2

        self.mask_predictor = MaskPredictor(
            input_dim=n_cols,
            hidden_dim=mask_predictor_hidden_dim,
            output_dim=n_cols,
        )

        self.apply(weights_init)

    def forward(self, x):
        if isinstance(x, tuple) and len(x) == 2:
            row_indices, col_indices = x
            reconstructed_values = self._forward_indices(row_indices, col_indices)
            unique_rows = row_indices.unique()
            full_reconstructed_matrix = self._reconstruct_rows(unique_rows)
            row_mapping = {row.item(): i for i, row in enumerate(unique_rows)}
            batch_indices = torch.tensor(
                [row_mapping[idx.item()] for idx in row_indices],
                device=row_indices.device,
            )
            batch_reconstructed_rows = full_reconstructed_matrix[batch_indices]
            predicted_mask_logits = self.mask_predictor(batch_reconstructed_rows)

            return reconstructed_values, predicted_mask_logits

        if x.dim() != 2:
            raise ValueError(
                f"Input to DMF should be (B, N) or (row_indices, col_indices), got {x.shape}"
            )

        batch_size = x.shape[0]
        reconstructed_matrix = torch.zeros_like(x)

        for i in range(batch_size):
            row_idx = torch.tensor([i % self.n_rows], device=x.device)
            reconstructed_row = self._reconstruct_rows(row_idx).squeeze(0)
            reconstructed_matrix[i] = reconstructed_row
        predicted_mask_logits = self.mask_predictor(reconstructed_matrix)

        return reconstructed_matrix, predicted_mask_logits

    def _forward_indices(self, row_indices, col_indices):
        row_emb = self.row_embedding(row_indices)
        col_emb = self.col_embedding(col_indices)
        combined_emb = torch.cat([row_emb, col_emb], dim=-1)
        reconstructed_values = self.reconstruction_mlp(combined_emb).squeeze(-1)
        return reconstructed_values

    def _reconstruct_rows(self, row_indices):
        device = row_indices.device
        batch_size = row_indices.size(0)
        col_indices = torch.arange(self.n_cols, device=device).repeat(batch_size, 1)
        row_indices_expanded = row_indices.unsqueeze(1).expand(-1, self.n_cols)
        flat_row_indices = row_indices_expanded.reshape(-1)
        flat_col_indices = col_indices.reshape(-1)
        flat_values = self._forward_indices(flat_row_indices, flat_col_indices)
        reconstructed_rows = flat_values.reshape(batch_size, self.n_cols)

        return reconstructed_rows

    def get_full_matrix(self):
        device = self.row_embedding.weight.device
        all_rows = torch.arange(self.n_rows, device=device)
        full_matrix = self._reconstruct_rows(all_rows)
        return full_matrix

    def get_imputed_data(self):
        self.eval()
        with torch.no_grad():
            full_matrix = self.model.get_full_matrix()
            imputed_data = self.full_data_tensor.clone()
            unobserved_mask = self.full_mask_tensor == 0
            imputed_data[unobserved_mask] = full_matrix[unobserved_mask]

        return imputed_data
