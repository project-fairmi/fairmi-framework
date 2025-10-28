from lightning import LightningModule
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
from safetensors.torch import load_file
from transformers import AutoModel, AutoImageProcessor
from open_clip import create_model_from_pretrained, get_tokenizer

class EmbeddingsExtractorModule(LightningModule):
    def __init__(self, model_id: str, pretrained: bool = False, weights_path: str = None, type: bool = False):
        super().__init__()
        self.model_id = model_id
        self.pretrained = pretrained
        self.weights_path = weights_path
        self.type = type
        if type == 'timm':
            self.model = self.configure_model()
            self.model.eval()
            self.transforms = self.configure_model_transform()
        elif type == 'clip':
            self.model, self.transforms = create_model_from_pretrained(model_id, cache_dir=weights_path)
            self.tokenizer = get_tokenizer(model_id, cache_dir=weights_path)
        elif type == 'huggingface':
            self.model = AutoModel.from_pretrained(model_id, cache_dir=weights_path)
            self.transforms = AutoImageProcessor.from_pretrained(model_id, cache_dir=weights_path)

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
        if self.type == 'timm':
            return self.model(x['image'])
        elif self.type == 'clip':
            return self.model.encode_image(x['image'])
        else:
            return self.model(x['image'])[1]
