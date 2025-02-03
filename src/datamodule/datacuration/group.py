import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from typing import List

class FeatureExtractor:
    """
    Classe responsável por extrair features de um dataset e aplicar clusterização.
    Não “sabe” detalhes de como o dataset é criado, apenas consome um Dataset PyTorch.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 32,
        n_clusters: int = 5,
        device: str = "cpu"
    ):
        """
        Args:
            model (torch.nn.Module): Modelo PyTorch para extrair features.
            batch_size (int, optional): Tamanho do batch para a extração.
            n_clusters (int, optional): Número de clusters no KMeans.
            device (str, optional): 'cpu' ou 'cuda'.
        """
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.device = device

    def extract_features(self, dataset: torch.utils.data.Dataset) -> torch.Tensor:
        """
        Extrai as features do dataset usando o modelo fornecido.
        """
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                # Convertemos para CPU para não acumular GPU se for grande
                all_features.append(outputs.cpu())

        # Concatena tudo em um único tensor
        features = torch.cat(all_features, dim=0)
        return features

    def cluster_features(self, features: torch.Tensor) -> List[int]:
        """
        Aplica KMeans nas features e retorna os rótulos de cluster.
        """
        # Flatten para 2D caso seja [N, ...]
        features_2d = features.view(features.size(0), -1)

        # Usa sklearn KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_2d.numpy())
        return labels.tolist()

    def collate_fn(self, batch):
        """
        Custom collate para empilhar as imagens em um tensor.
        """
        images = torch.stack([item['image'] for item in batch])
        # Podemos retornar o batch inteiro, mas aqui só o que é necessário.
        return {'image': images}