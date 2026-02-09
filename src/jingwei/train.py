import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .dcae import DCAEImputer
from .datasets import CSVDataset
from .dmf import DMFImputer


class ImputationTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.setup_directories()

    def setup_directories(self):
        self.results_dir = Path(self.args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.log_dir = self.results_dir / "logs"
        self.output_dir = self.results_dir / "outputs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = Path(self.args.data_path).stem
        self.run_dir = self.output_dir / f"{self.args.method}_{dataset_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, dataset):
        data_tensor = dataset.data_normalized
        mask_tensor = dataset.mask

        common_params = {
            "full_data_tensor": data_tensor,
            "full_mask_tensor": mask_tensor,
            "batch_size": self.args.batch_size,
        }

        if self.args.method.upper() == "DMF":
            model = DMFImputer(
                **common_params,
                embedding_dim=self.args.embedding_dim,
                hidden_dims=self.args.hidden_dims,
                mask_weight=self.args.mask_weight,
                reconstruction_weight=self.args.reconstruction_weight,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.method.upper() == "DCAE":
            model = DCAEImputer(
                **common_params,
                ae_dim=self.args.latent_dim,
                mask_predictor_hidden_dim=self.args.hidden_dims[0]
                if self.args.hidden_dims
                else 128,
                lambda_mask=self.args.mask_weight,
                num_encoder_blocks=self.args.num_encoder_blocks,
                num_decoder_blocks=self.args.num_decoder_blocks,
                dilation=self.args.dilation,
                learning_rate=self.args.learning_rate,
            )
        else:
            raise ValueError(f"Unknown method: {self.args.method}")

        return model

    def setup_device(self):
        if self.args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_id)
            self.device = "cuda"
            self.devices = 1
            print(f"Using GPU {self.args.gpu_id}")
        elif self.args.device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.devices = 1
                print("Using CUDA device")
            else:
                print("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.devices = "auto"
        elif self.args.device == "cpu":
            self.device = "cpu"
            self.devices = "auto"
            print("Using CPU")
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.devices = 1
                print("Auto-detected CUDA device")
            else:
                self.device = "cpu"
                self.devices = "auto"
                print("Auto-detected CPU")

    def setup_callbacks(self, dataset_name):
        callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=(
                f"{self.args.method}_{dataset_name}_mask{self.args.mask_weight:.2f}_"
                "{epoch:02d}_{train/loss:.4f}"
            ),
            monitor="train/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        if self.args.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="train/loss",
                patience=self.args.patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stop_callback)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        return callbacks

    def train(self):
        dataset_path = self.args.data_path
        dataset = CSVDataset(dataset_path)
        dataset_name = Path(dataset_path).stem

        print(f"\n{'=' * 50}")
        print(f"Training {self.args.method.upper()} on dataset: {dataset_name}")
        print(f"{'=' * 50}")
        print(f"Shape: {dataset.data_normalized.shape}")
        print(f"Missing rate: {1 - dataset.mask.mean().item():.2%}")
        print(f"Mask weight: {self.args.mask_weight:.2f}")
        print(f"Results will be saved to: {self.run_dir}")
        print(f"{'=' * 50}\n")

        model = self.get_model(dataset)

        logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=f"{self.args.method}_{dataset_name}",
            version=f"mask_{self.args.mask_weight:.2f}",
        )

        callbacks = self.setup_callbacks(dataset_name)

        if self.args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.args.device

        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=getattr(self.args, "progress_bar", False),
            log_every_n_steps=min(self.args.log_interval, len(model.train_dataloader())),
            accelerator=self.device,
            devices=self.devices,
            precision=32,
            gradient_clip_val=self.args.gradient_clip
            if hasattr(self.args, "gradient_clip")
            else None,
        )
        trainer.fit(model)

        self.save_config(dataset_name)
        self.save_imputed_data(model, dataset, dataset_name)
        print(f"\nTraining completed. Logs saved to: {logger.log_dir}")
        print(f"To view logs: tensorboard --logdir={self.log_dir}")
        print(f"Imputed data saved to: {self.run_dir}")

        return model

    def save_config(self, dataset_name):
        import json

        config = vars(self.args)
        config["dataset_name"] = dataset_name
        config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

    def save_imputed_data(self, model, dataset, dataset_name):
        model.eval()
        with torch.no_grad():
            normalized_imputed = model.get_imputed_data()
            imputed_data = dataset.inverse_transform(normalized_imputed).cpu().numpy()
            df_original = pd.read_csv(self.args.data_path)
            if (
                df_original.shape[0] == imputed_data.shape[0]
                and df_original.shape[1] - 1 == imputed_data.shape[1]
            ):
                df_imputed = pd.DataFrame(imputed_data)
                df_imputed.insert(0, df_original.columns[0], df_original.iloc[:, 0].values)
                df_imputed.columns = df_original.columns
            else:
                df_imputed = pd.DataFrame(imputed_data)

        output_path = self.run_dir / f"{dataset_name}_imputed.csv"
        df_imputed.to_csv(output_path, index=False)

        mask = dataset.mask.cpu().numpy()
        mask_df = pd.DataFrame(mask)
        mask_path = self.run_dir / f"{dataset_name}_mask.csv"
        mask_df.to_csv(mask_path, index=False)

        print(f"Imputed data saved to: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train imputation models for PRISM")
    parser.add_argument(
        "--method",
        type=str,
        choices=["DMF", "DCAE"],
        default="DMF",
        help="Imputation method to use",
    )

    parser.add_argument("--data-path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[256, 128], help="Hidden layer dimensions"
    )

    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension for DMF")

    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension for DCAE (ae_dim)")
    parser.add_argument(
        "--num-encoder-blocks", type=int, default=2, help="Number of encoder blocks for DCAE"
    )
    parser.add_argument(
        "--num-decoder-blocks", type=int, default=2, help="Number of decoder blocks for DCAE"
    )
    parser.add_argument("--dilation", type=int, default=2, help="Dilation factor for DCAE")

    parser.add_argument(
        "--mask-weight",
        type=float,
        default=0.5,
        help="Weight for mask prediction loss (lambda_mask for DCAE)",
    )
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )

    parser.add_argument("--max-epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")

    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")

    parser.add_argument(
        "--results-dir", type=str, default="./results", help="Directory for saving all results"
    )
    parser.add_argument("--log-interval", type=int, default=50, help="Logging interval in steps")
    parser.add_argument("--progress-bar", action="store_true", help="Show progress bar during training")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training",
    )
    parser.add_argument("--gpu-id", type=int, default=None, help="Specific GPU ID to use (0, 1, etc.)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    pl.seed_everything(114514)
    trainer = ImputationTrainer(args)
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
