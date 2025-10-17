import os
import torch
import umap
from sklearn.manifold import TSNE
from src.model.extractor.embeddings import EmbeddingsExtractorModule
from src.datamodule.dataset.chest import CheXpertModule
from src.datamodule.dataset.natural import CelebAModule
from src.datamodule.dataset.skin import Ham10000Module
from src.config import config

DATASET_MODULES = {
    "chexpert": CheXpertModule,
    # "ham10000": Ham10000Module,
    # "celeba": CelebAModule
}

def load_embeddings(path):
    return torch.load(path)

def save_embeddings(path, embeddings, gender, age):
    torch.save({'embeddings': embeddings, 'gender': gender, 'age': age}, path)

def save_umap(path, umap_embeddings, gender, age):
    torch.save({'umap_embeddings': umap_embeddings, 'gender': gender, 'age': age}, path)

def save_tsne(path, tsne_embeddings, gender, age):
    torch.save({'tsne_embeddings': tsne_embeddings, 'gender': gender, 'age': age}, path)

def compute_umap(embeddings):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(embeddings.numpy())
    return torch.tensor(umap_emb)

def compute_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings.numpy())
    return torch.tensor(tsne_emb)

def extract_embeddings(extractor, datamodule):
    embeddings, gender, age = [], [], []
    for batch in datamodule.test_dataloader():
        with torch.no_grad():
            embeddings.append(extractor(batch).cpu())
            gender.append(batch['gender'].cpu())
            age.append(batch['age'].cpu())
    return torch.cat(embeddings), torch.cat(gender), torch.cat(age)

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
        data = load_embeddings(embeddings_path)
        embeddings, gender, age = data['embeddings'], data['gender'], data['age']
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
            num_groups=3
        )
        datamodule.setup(stage='test')
        embeddings, gender, age = extract_embeddings(extractor, datamodule)
        save_embeddings(embeddings_path, embeddings, gender, age)

    if not os.path.exists(umap_path):
        umap_embeddings = compute_umap(embeddings)
        save_umap(umap_path, umap_embeddings, gender, age)
    else:
        print(f"UMAP embeddings already exist for {dataset_name}.")

    if not os.path.exists(tsne_path):
        tsne_embeddings = compute_tsne(embeddings)
        save_tsne(tsne_path, tsne_embeddings, gender, age)
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
