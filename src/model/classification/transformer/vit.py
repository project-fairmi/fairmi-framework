from ..base import ClassificationModel
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
import torch

class VisionTransformerModel(ClassificationModel):
    def __init__(self, model_id: str = 'timm/vit_small_patch14_reg4_dinov2.lvd142m', cache_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
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
            num_classes=self.num_classes
        )
        return model
    
    def configure_optimizers(self):
        """Creates the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        # Create the optimizer for the model
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # Create the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
