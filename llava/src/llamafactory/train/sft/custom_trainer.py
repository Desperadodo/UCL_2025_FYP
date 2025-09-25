import torch
from typing import Optional, Dict, List, Any
import gc

from .trainer import CustomSeq2SeqTrainer


class _ZeroImageDataLoader:
    """
    A wrapper for a dataloader that zeros out pixel_values in each batch.
    It is designed to be lightweight and not interfere with other dataloader attributes.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            # Clean sanity: keep placeholders but inject zero visual features by flag
            batch["force_zero_image_features"] = True
            yield batch

    def __len__(self) -> int:
        return len(self.dataloader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataloader, name)


class SanityCheckTrainer(CustomSeq2SeqTrainer):
    """
    A custom Trainer that performs a sanity check evaluation with zeroed-out image inputs
    immediately after each standard evaluation.
    It inherits from CustomSeq2SeqTrainer to ensure all custom functionalities are preserved.
    """
    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overrides the default evaluate method to add a sanity check run.
        """
        # 1. Run the standard evaluation first.
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 2. Run the sanity check evaluation with zeroed images.
        if self.is_world_process_zero():
            print("\n--- Running Sanity Check: Evaluating with zeroed images ---")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        sanity_dataloader = _ZeroImageDataLoader(eval_dataloader)
        sanity_metric_prefix = f"{metric_key_prefix}_sanity_check"

        sanity_output = self.evaluation_loop(
            sanity_dataloader,
            description="Sanity Check",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=sanity_metric_prefix,
        )

        # Prefer a distinct key to avoid overriding the standard eval_loss
        # If upstream only accepts eval_* keys, keep the metric_key_prefix behavior and also mirror into a custom key
        sanity_loss = sanity_output.metrics["eval_sanity_check_loss"]

        log_dict = {
            "eval_sanity_check_loss": sanity_loss,
            "loss_sanity_check": sanity_loss,
        }

        if self.is_world_process_zero():
            self.control.should_log = True
            self.log(log_dict)
            self.control.should_log = False

        metrics.update(log_dict)
        return metrics
