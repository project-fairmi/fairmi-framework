import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from typing import Optional

class GroupSensitiveAttribute:
    
    def __init__(self, datamodule: pl.LightningDataModule, model: torch.nn.Module):
        self.datamodule = datamodule
        self.model = model
        self.features: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None

    def extract_features(self) -> torch.Tensor:
        self.model.eval()
        self.datamodule.setup(stage='test')
        dataloader = self.datamodule.test_dataloader()
        features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                outputs = self.model(inputs)
                features.append(outputs)

        self.features = torch.cat(features, dim=0)
        return self.features

    def cluster_features(self, eps: float = 0.5, min_samples: int = 5) -> torch.Tensor:
        if self.features is None:
            raise ValueError("Features have not been extracted. Call extract_features() first.")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = torch.tensor(dbscan.fit_predict(self.features.cpu().numpy()))
        return self.labels

    def add_cluster_labels_to_datamodule(self):
        if self.labels is None:
            raise ValueError("Clusters have not been computed. Call cluster_features() first.")
        
        self.datamodule.set_cluster_labels(self.labels)