import os
import torch
import umap
from torchdr import InfoTSNE
from sklearn.manifold import TSNE
from src.model.extractor.embeddings import EmbeddingsExtractorModule
from src.datamodule.dataset.chest import CheXpertModule
from src.datamodule.dataset.natural import CelebAModule
from src.datamodule.dataset.skin import Ham10000Module
from src.projects.group_similarity import config

DATASET_MODULES = {
    "chexpert": CheXpertModule,
    # "ham10000": Ham10000Module,
    # "celeba": CelebAModule
}

def load_embeddings(path):
    return torch.load(path)

def save_embeddings(path, data):
    torch.save(data, path)

def compute_infotsne(embeddings, n_components=2):
    info_tsne = InfoTSNE(n_components=n_components, random_state=42, device='cuda')
    info_tsne_emb = info_tsne.fit_transform(embeddings)
    return torch.tensor(info_tsne_emb)

def extract_embeddings(extractor, datamodule):
    all_data = {} # Initialize as empty dictionary
    for batch in datamodule.test_dataloader():
        with torch.no_grad():
            extracted_embeddings = extractor(batch)
            
            # Initialize 'embeddings' key if it doesn't exist
            if 'embeddings' not in all_data:
                all_data['embeddings'] = []
            all_data['embeddings'].append(extracted_embeddings)

            for key, value in batch.items():
                if key == 'image':
                    continue
                if key == 'group':
                    for group_key, group_value in value.items():
                        # Ensure group_key is initialized
                        if group_key not in all_data:
                            all_data[group_key] = []
                        all_data[group_key].append(group_value.cpu())
                    continue    
                # Ensure other batch keys are initialized
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(value.cpu())

    for key in all_data:
        all_data[key] = torch.cat(all_data[key])
    return all_data

def process_dataset(dataset_name, extractor, dimmensions):
    output_dir = f"output/embeddings/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    model = config['model']['name']
    embeddings_path = os.path.join(output_dir, f'{dataset_name}_{model}_embeddings.pt')

    if os.path.exists(embeddings_path):
        print(f"Loading embeddings for {dataset_name}, computing missing embeddings...")
        all_extracted_data = load_embeddings(embeddings_path)
    else:
        print(f"Extracting embeddings for {dataset_name}...")
        datamodule_class = DATASET_MODULES[dataset_name]
        datamodule = datamodule_class(
            data_dir=config['data'][dataset_name]['data_path'],
            image_data_dir=config['data'][dataset_name]['image_data_path'],
            batch_size=config['training']['batch_size'],
            model_transform=extractor.transforms,
            augment_train=False,
            fraction=config['training']['fraction'],
            num_workers=config['data'][dataset_name]['num_workers'],
            task=config['data'][dataset_name]['task'],
            num_groups=config['data'][dataset_name]['num_groups']
        )
        datamodule.setup(stage='test')
        all_extracted_data = extract_embeddings(extractor, datamodule)
        save_embeddings(embeddings_path, all_extracted_data)

    for n_components in dimmensions:
        if all_extracted_data.get(n_components):
            print(f"Skipping {dataset_name}: embeddings for n_components={n_components} found.")
        else:
            embeddings_n_components = compute_infotsne(all_extracted_data['embeddings'], n_components=n_components)
            all_extracted_data[n_components] = embeddings_n_components

def main():
    model = config['model']['name']
    model_id = config['model'][model]['id']
    weights_path = config['model'][model]['weights_path']
    type = config['model'][model]['type']
    extractor = EmbeddingsExtractorModule(model_id=model_id, pretrained=True, weights_path=weights_path, type=type)

    for dataset_name in DATASET_MODULES:
        process_dataset(dataset_name, extractor, dimmensions=config['groups']['embeddings_dimmension'])

if __name__ == "__main__":
    main()