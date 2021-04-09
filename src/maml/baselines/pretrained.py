import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import EarlyStopping


from pathlib import Path
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score

from ..lstm import Classifier
from .data import PretrainingData
from src.utils import set_seed

from typing import Dict, Tuple, Type, Any, Optional


def train_model(model: pl.LightningModule, hparams: Namespace) -> pl.LightningModule:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=hparams.patience, verbose=True, mode="min",
    )
    trainer = pl.Trainer(
        default_save_path=model.version_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)

    return model


class Pretrainer(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        set_seed()

        self.hparams = hparams
        self.data_folder = Path(hparams.data_folder)
        self.version_folder = (
            self.data_folder / "maml_models" / f"version_{self.hparams.model_version}"
        )

        # load the model info
        with (self.version_folder / "model_info.json").open("r") as f:
            self.model_info = json.load(f)

        self.classifier = Classifier(
            input_size=self.model_info["input_size"],
            classifier_vector_size=self.model_info["classifier_vector_size"],
            classifier_dropout=self.model_info["classifier_dropout"],
            classifier_base_layers=self.model_info["classifier_base_layers"],
            num_classification_layers=self.model_info["num_classification_layers"],
        )

        self.best_val_loss: float = np.inf

        self.loss = nn.BCELoss()

        self.normalizing_dict = self.get_dataset(subset="training").normalizing_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.learning_rate,)

    def get_dataset(
        self, subset: str, cache: bool = True, norm_dict: Optional[Dict] = None
    ) -> PretrainingData:
        return PretrainingData(
            self.data_folder,
            subset,
            self.model_info["remove_b1_b10"],
            cache=cache,
            normalizing_dict=norm_dict,
            include_sudan_ethiopia=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training", norm_dict=None),
            shuffle=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="validation", norm_dict=self.normalizing_dict),
            batch_size=self.hparams.batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def test_dataloader(self):
        return DataLoader(self.get_dataset(subset="testing",), batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.forward(x)
        loss = self.loss(preds.squeeze(1), y)

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.forward(x)
        loss = self.loss(preds.squeeze(1), y)

        return {"val_loss": loss, "log": {"val_loss": loss}, "preds": preds, "labels": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_auc_roc": roc_auc_score(labels, preds),
            "val_accuracy": accuracy_score(labels, preds > 0.5),
        }

        if float(avg_loss) < self.best_val_loss:
            self.best_val_loss = float(avg_loss)
            print("Saving best state_dict")
            self.save_state_dict()

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def save_state_dict(self) -> None:
        torch.save(self.classifier.state_dict(), self.version_folder / "pretrained.pth")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            # assumes this is being run from "scripts"
            "--data_folder": (str, str(Path("../data"))),
            "--model_version": (int, 0),
            "--max_epochs": (int, 1000),
            "--learning_rate": (float, 0.001),
            "--patience": (int, 10),
            "--batch_size": (int, 64),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        return parser
