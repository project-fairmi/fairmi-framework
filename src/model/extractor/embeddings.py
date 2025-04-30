from lightning import LightningModule
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
from safetensors.torch import load_file

class EmbeddingsExtractorModule(LightningModule):
    def __init__(self, model_id: str, pretrained: bool = False, weights_path: str = None):
        super().__init__()
        self.model_id = model_id
        self.pretrained = pretrained
        self.weights_path = weights_path
        self.model = self.configure_model()
        self.model.eval()
        self.transforms = self.configure_model_transform()

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
            num_classes=0,
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

    def forward(self, x):
        image = self.transforms(x['image'])
        return self.model(image)
