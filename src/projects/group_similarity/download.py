from transformers import AutoModel, AutoImageProcessor
from open_clip import create_model_from_pretrained, get_tokenizer

# model_name = "microsoft/rad-dino"
model_name = "facebook/dinov2-base"
cache_directory = "/scratch/unifesp/fairmi/dilermando.queiroz/fairmi-framework/.cache"

model = AutoModel.from_pretrained(model_name, cache_dir=cache_directory)
processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_directory)

# Load the model and config files from the Hugging Face Hub
# model_clip, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
#                                                        cache_dir=cache_directory)
# tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
#                            cache_dir=cache_directory)