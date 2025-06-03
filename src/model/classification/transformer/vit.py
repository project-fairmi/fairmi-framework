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
        freeze_layers: int = 0,  # Parameter to specify number of layers to freeze
        **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.pretrained = pretrained
        self.max_epochs = max_epochs
        self.weights_path = weights_path
        self.freeze_layers = freeze_layers  # Store the parameter
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
            drop_path_rate=0,
            drop_rate=0
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
        
        # Handle layer freezing based on freeze_layers parameter
        if self.freeze_layers > 0:
            # Freeze all parameters first
            for param in model.parameters():
                param.requires_grad = False

            if hasattr(model, 'blocks'):
                # For Vision Transformers, the blocks are typically named 'block.N'
                num_blocks = len(model.blocks)

                if self.freeze_layers == 1:
                    # Only unfreeze the classification head
                    for param in model.head.parameters():
                        param.requires_grad = True
                else:
                    # Unfreeze the classification head and the last (freeze_layers - 1) blocks
                    layers_to_unfreeze = min(self.freeze_layers - 1, num_blocks)

                    # Unfreeze the classification head
                    for param in model.head.parameters():
                        param.requires_grad = True

                    # Unfreeze the specified number of blocks from the end
                    for i in range(num_blocks - layers_to_unfreeze, num_blocks):
                        for param in model.blocks[i].parameters():
                            param.requires_grad = True

            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            print(f"Trainable parameters: {trainable_params}")

        return model

    def configure_optimizers(self):
        """Creates the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        beta_2 = 0.999
        # Create the optimizer for the model
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, beta_2),
        )



        # --- PyTorch Linear Warmup + Cosine Annealing Scheduler ---
        # Configuration based on found LR from lr_find
        warmup_epochs = int(0.1 * self.max_epochs)
        total_epochs = self.max_epochs
        peak_lr = self.learning_rate 
        initial_lr = peak_lr / 100.0
        eta_min = peak_lr / 50.0

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
