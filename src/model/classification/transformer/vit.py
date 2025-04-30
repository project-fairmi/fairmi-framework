from ..base import ClassificationModel
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from safetensors.torch import load_file

class VisionTransformerModel(ClassificationModel):
    def __init__(self,
        model_id: str = 'timm/vit_small_patch14_reg4_dinov2.lvd142m',
        pretrained: bool = False,
        max_epochs: int = 50,
        weights_path: str = None,
        **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.pretrained = pretrained
        self.max_epochs = max_epochs
        self.weights_path = weights_path
        self.model = self.configure_model()
        self.transform = self.configure_model_transform()
        self.save_hyperparameters()
    
    def configure_model_transform(self):
        """Configures the dataloader for the model.
        """
        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        return transform

    def configure_model(self):
        """Configures the model architecture.
        """
        model = timm.create_model(
            self.model_id,
            pretrained=False,
            num_classes=self.num_classes,
        )

        if self.pretrained:
            state_dict = load_file(self.weights_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dictionary from '{self.weights_path}'.")

            report_lines = []
            if missing_keys:
                report_lines.append(f"  - WARNING: Missing keys ({len(missing_keys)}): {missing_keys}")

            if unexpected_keys:
                report_lines.append(f"  - WARNING: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
                # Keep the helpful note about classification head
                report_lines.append("    (Note: Unexpected keys often occur in the classification head when num_classes differs)")

            if report_lines:
                print("State dictionary loading issues found:")
                print("\n".join(report_lines))
            else:
                # Clear confirmation if everything matches
                print("State dictionary loaded successfully with no key mismatches.")

        return model
    
    def configure_optimizers(self):
        """Creates the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        # Create the optimizer for the model
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # --- PyTorch Linear Warmup + Cosine Annealing Scheduler ---
        # Configuration based on found LR from lr_find
        warmup_epochs = 5  # Keep the original warmup duration (adjust if max_epochs is very short/long)
        total_epochs = self.max_epochs
        peak_lr = self.learning_rate # Target LR after warmup (should be ~3.02e-6 based on lr_find)
        initial_lr = peak_lr / 100.0 # Start warmup at 1/100th of peak LR
        eta_min = peak_lr / 100.0    # Allow annealing down to 1/100th of peak LR

        print(f"Scheduler Config: peak_lr={peak_lr:.2e}, initial_lr={initial_lr:.2e}, eta_min={eta_min:.2e}, warmup={warmup_epochs}, total={total_epochs}")

        warmup_scheduler = LinearLR(optimizer,
                                   start_factor=initial_lr / peak_lr if peak_lr > 0 else 0,
                                   total_iters=warmup_epochs)

        cosine_scheduler = CosineAnnealingLR(optimizer,
                                           T_max=(total_epochs - warmup_epochs),
                                           eta_min=eta_min)

        scheduler = SequentialLR(optimizer,
                               schedulers=[warmup_scheduler, cosine_scheduler],
                               milestones=[warmup_epochs])
        # ---------------------------------------------------------

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
