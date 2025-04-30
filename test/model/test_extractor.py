import pytest
import torch
import os
from src.model.extractor.embeddings import EmbeddingsExtractorModule
from src.datamodule.dataset.vision.brset import Brset
from src.config import config

@pytest.fixture
def sample_batch():
    """Generate a sample batch of random image tensors."""
    return {"image": torch.randn(2, 3, 224, 224)}  # Batch of 2 RGB images

@pytest.mark.parametrize("model_name,expected_features", [
    ("dino-small", 384),
    ("dino-base", 768),
])
def test_forward_pass(sample_batch, model_name, expected_features):
    """Test basic forward pass with DINO model architectures."""
    model_config = config['model'][model_name]
    model = EmbeddingsExtractorModule(
        model_id=model_config['id'],
        pretrained=config['model']['pretrained'],
        weights_path=model_config['weights_path']
    )
    outputs = model(sample_batch)
    
    # Verify output shape and type
    assert isinstance(outputs, torch.Tensor)
    assert outputs.ndim == 2
    assert outputs.shape[0] == 2  # Batch size preserved
    assert outputs.shape[1] == expected_features  # Feature dimension matches expected size

@pytest.mark.parametrize("model_name", ["dino-small", "dino-base"])
def test_transform_configuration(model_name):
    """Test that DINO model transforms are properly configured."""
    model_config = config['model'][model_name]
    model = EmbeddingsExtractorModule(
        model_id=model_config['id'],
        pretrained=config['model']['pretrained'],
        weights_path=model_config['weights_path']
    )
    transforms = model.configure_model_transform()
    
    assert transforms is not None
    assert len(transforms.transforms) > 2  
    
    transform_names = [t.__class__.__name__ for t in transforms.transforms]
    assert any("Resize" in name for name in transform_names)
    assert any("Normalize" in name for name in transform_names)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_inference():
    """Test model works on GPU if available."""
    model_config = config['model']['dino-small']
    model = EmbeddingsExtractorModule(
        model_id=model_config['id'],
        pretrained=config['model']['pretrained'],
        weights_path=model_config['weights_path']
    ).to("cuda")
    batch = {"image": torch.randn(2, 3, 224, 224).to("cuda")}
    outputs = model(batch)
    
    assert outputs.is_cuda
    assert outputs.shape[0] == 2
    assert outputs.ndim == 2