import os
import torch
import umap
from sklearn.manifold import TSNE
from src.model.extractor.embeddings import EmbeddingsExtractorModule
from src.datamodule.dataset.chest import CheXpertModule
from src.datamodule.dataset.natural import CelebAModule
from src.datamodule.dataset.skin import Ham10000Module
from src.projects.group_similarity import config

DATASET_MODULES = {
    "chexpert": CheXpertModule,
    "ham10000": Ham10000Module,
    "celeba": CelebAModule
}

def load_embeddings(path):
    return torch.load(path)

def save_embeddings(path, data):
    torch.save(data, path)

def save_umap(path, data):
    torch.save(data, path)

def save_tsne(path, data):
    torch.save(data, path)

def compute_umap(embeddings):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(embeddings.numpy())
    return torch.tensor(umap_emb)

def compute_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings.numpy())
    return torch.tensor(tsne_emb)

def extract_embeddings(extractor, datamodule):
    all_data = {}
    for batch in datamodule.test_dataloader():
        with torch.no_grad():
            extracted_embeddings = extractor(batch).cpu()
            all_data['embeddings'].append(extracted_embeddings)

            for key, value in batch.items():
                if key == 'image':
                    continue
                if key == 'group':
                    for group_key, group_value in value.items():
                        all_data[group_key].append(group_value.cpu())
                    continue    
                all_data[key].append(value.cpu())

    for key in all_data:
        all_data[key] = torch.cat(all_data[key])
    return all_data

def process_dataset(dataset_name, extractor):
    output_dir = f"output/embeddings/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    model = config['model']['name']
    embeddings_path = os.path.join(output_dir, f'{dataset_name}_{model}_embeddings.pt')
    umap_path = os.path.join(output_dir, f'{dataset_name}_{model}_umap.pt')
    tsne_path = os.path.join(output_dir, f'{dataset_name}_{model}_tsne.pt')

    if os.path.exists(embeddings_path):
        if os.path.exists(umap_path) and os.path.exists(tsne_path):
            print(f"Skipping {dataset_name}: embeddings, UMAP and t-SNE found.")
            return
        print(f"Loading embeddings for {dataset_name}, computing missing embeddings...")
        all_extracted_data = load_embeddings(embeddings_path)
        embeddings = all_extracted_data['embeddings']
    else:
        print(f"Extracting embeddings for {dataset_name}...")
        datamodule_class = DATASET_MODULES[dataset_name]
        datamodule = datamodule_class(
            data_dir=config['data'][dataset_name]['data_path'],
            image_data_dir=config['data'][dataset_name]['image_data_path'],
            batch_size=config['training']['batch_size'],
            model_transform=False,
            augment_train=False,
            fraction=config['training']['fraction'],
            num_workers=config['data'][dataset_name]['num_workers'],
            task=config['data'][dataset_name]['task'],
            num_groups=config['data'][dataset_name]['num_groups']
        )
        datamodule.setup(stage='test')
        all_extracted_data = extract_embeddings(extractor, datamodule)
        save_embeddings(embeddings_path, all_extracted_data)
        embeddings = all_extracted_data['embeddings']

    if not os.path.exists(umap_path):
        umap_embeddings = compute_umap(embeddings)
        save_umap(umap_path, {'umap_embeddings': umap_embeddings, **{k: v for k, v in all_extracted_data.items() if k != 'embeddings'}})
    else:
        print(f"UMAP embeddings already exist for {dataset_name}.")

    if not os.path.exists(tsne_path):
        tsne_embeddings = compute_tsne(embeddings)
        save_tsne(tsne_path, {'tsne_embeddings': tsne_embeddings, **{k: v for k, v in all_extracted_data.items() if k != 'embeddings'}})
    else:
        print(f"t-SNE embeddings already exist for {dataset_name}.")

def main():
    model = config['model']['name']
    model_id = config['model'][model]['id']
    weights_path = config['model'][model]['weights_path']
    extractor = EmbeddingsExtractorModule(model_id=model_id, pretrained=True, weights_path=weights_path)

    for dataset_name in DATASET_MODULES:
        process_dataset(dataset_name, extractor)

if __name__ == "__main__":
    main()
