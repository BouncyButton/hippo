import json
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_5epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainer_1epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1


class nnUNetTrainerSmoke(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """Minimal end-to-end trainer for local smoke tests."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 1
        self.num_val_iterations_per_epoch = 1


class nnUNetTrainer_10epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10


class nnUNetTrainer_20epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20


class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainer_50epochsEarlyStopping(nnUNetTrainer):
    """50-epoch trainer with conservative validation-Dice early stopping.

    The online foreground Dice is evaluated after every standard nnU-Net epoch
    (250 optimizer updates). Training stops after epoch 15 when Dice has failed
    to improve by at least 0.001 for 10 consecutive epochs. The ordinary
    checkpoint_best.pth behavior is preserved, while checkpoint_final.pth
    records the actual stopping point.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.early_stopping_min_epochs = 15
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 1e-3
        self._early_stopping_best_dice = -np.inf
        self._early_stopping_bad_epochs = 0
        self._early_stopping_requested = False
        self._early_stopping_epoch = None

    def on_epoch_end(self):
        completed_epoch = self.current_epoch + 1
        current_dice = float(self.logger.my_fantastic_logging['mean_fg_dice'][-1])

        if current_dice > self._early_stopping_best_dice + self.early_stopping_min_delta:
            self._early_stopping_best_dice = current_dice
            self._early_stopping_bad_epochs = 0
        else:
            self._early_stopping_bad_epochs += 1

        super().on_epoch_end()

        self.print_to_log_file(
            "early_stopping",
            {
                "monitor": "mean_fg_dice",
                "best": round(self._early_stopping_best_dice, 6),
                "bad_epochs": self._early_stopping_bad_epochs,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "min_epochs": self.early_stopping_min_epochs,
            },
        )

        if (
            completed_epoch >= self.early_stopping_min_epochs
            and self._early_stopping_bad_epochs >= self.early_stopping_patience
        ):
            self._early_stopping_requested = True
            self._early_stopping_epoch = completed_epoch
            self.print_to_log_file(
                f"Early stopping at epoch {completed_epoch}: validation foreground Dice "
                f"did not improve by {self.early_stopping_min_delta} for "
                f"{self.early_stopping_patience} epochs.",
                also_print_to_console=True,
            )

    def run_training(self):
        self.on_train_start()

        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self._early_stopping_requested:
                break

        self.on_train_end()
        if self.local_rank == 0:
            summary = {
                "max_epochs": self.num_epochs,
                "epochs_completed": len(self.logger.my_fantastic_logging['train_losses']),
                "stopped_early": self._early_stopping_requested,
                "stopping_epoch": self._early_stopping_epoch,
                "monitor": "mean_fg_dice",
                "best_dice_with_min_delta": self._early_stopping_best_dice,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "min_epochs": self.early_stopping_min_epochs,
            }
            output_path = Path(self.output_folder) / "early_stopping_summary.json"
            output_path.write_text(json.dumps(summary, indent=2) + "\n")


class nnUNetTrainer_100epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainer_500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_750epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainer_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000

    
class nnUNetTrainer_4000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4000


class nnUNetTrainer_8000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 8000
