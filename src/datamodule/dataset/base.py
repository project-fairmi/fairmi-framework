from typing import List, Type, Optional, Dict, Any
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import v2 as transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import re
import numpy as np
from torch.utils.data import default_collate
from PIL import Image

RANDOM_SEED = 42
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.1
TEST_SPLIT = 0.3

class Dataset(TorchDataset):
    """Base dataset for loading and processing data for machine learning tasks."""

    def __init__(self, data_dir: str, image_data_dir: str, labels_file: str, image_column: str, type: str, 
                 transform: bool = False, fraction: float = 1, age_column: Optional[str] = None, 
                 gender_column: Optional[str] = None, num_groups: int = 4, task: Optional[str] = None, 
                 patient_id_column: Optional[str] = None):
        """Initializes the dataset."""
        random.seed(RANDOM_SEED)
        self.image_data_dir = image_data_dir
        self.type = type
        self.transform = transform
        self.data_dir = data_dir
        self.fraction = fraction
        self.age_column = age_column
        self.gender_column = gender_column
        self.num_groups = num_groups
        self.task = task
        self.patient_id_column = patient_id_column
        self.image_column = image_column
        
        self.labels = self._set_labels(labels_file)
        self.initial_transform = self._get_initial_transform()

    def _set_labels(self, labels_file: str) -> pd.DataFrame:
        """Sets the labels for the dataset."""
        if labels_file.endswith('.csv'):
            return pd.read_csv(f"{self.data_dir}/{labels_file}")
        else:
            raise ValueError(f'Invalid file format: {labels_file} (must be .csv)')

    def _get_initial_transform(self) -> transforms.Compose:
        """Returns the initial transformation pipeline for the dataset."""
        return transforms.Compose([
            transforms.ToImage(),  
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    def configure_dataset(self):
        """Configures the dataset by creating age and gender groups, and preparing labels based on the task."""
        if self.age_column:
            self.labels = self.labels.dropna(subset=[self.age_column])
            self.labels['age_group'] = self._create_age_groups(self.labels[self.age_column])
        else:
            self.labels['age_group'] = np.nan
        
        if self.gender_column:
            self.labels = self.labels.dropna(subset=[self.gender_column])
            self.labels['gender_group'] = self._convert_gender_to_binary(self.labels[self.gender_column])
        else:
            self.labels['gender_group'] = np.nan
        
        if 'group' not in self.labels.columns:
            self.labels['group'] = np.nan

        if self.task:
            self.labels[self.task] = self.labels[self.task].fillna(0)
            self.labels['labels'] = self.labels[self.task]

    def split(self):
        """Splits the dataset into training, validation, and test sets based on patient IDs."""
        np.random.seed(RANDOM_SEED)

        patients = self.labels[self.patient_id_column].unique()
        np.random.shuffle(patients)

        train_size = int(TRAIN_SPLIT * len(patients))
        validation_size = int(VAL_SPLIT * len(patients))
        
        train_patients = patients[:train_size]
        remaining_patients = patients[train_size:]

        validation_patients = remaining_patients[:validation_size]
        test_patients = remaining_patients[validation_size:]

        if self.type == 'train':    
            self.labels = self.labels[self.labels[self.patient_id_column].isin(train_patients)].reset_index(drop=True)
            self.labels = self.labels.sample(frac=self.fraction).reset_index(drop=True)
        elif self.type == 'val':
            self.labels = self.labels[self.labels[self.patient_id_column].isin(validation_patients)].reset_index(drop=True)
        elif self.type in ['test', 'eval']:
            self.labels = self.labels[self.labels[self.patient_id_column].isin(test_patients)].reset_index(drop=True)
        else:
            raise ValueError(f'Invalid type: {self.type} (must be train, val or test/eval)')

    def transforms(self) -> transforms.Compose:
        """Returns the transformation pipeline for the dataset."""
        return transforms.Compose([
            self.initial_transform,
            transforms.RandAugment(num_ops=4)
        ])

    def _convert_gender_to_binary(self, column: pd.Series) -> pd.Series:
        """Converts gender values to binary format (0 for female, 1 for male)."""
        female_pattern = re.compile(r'^(f|feminine|female|woman|girl)$', re.IGNORECASE)
        male_pattern = re.compile(r'^(m|masculine|male|man|boy)$', re.IGNORECASE)

        return column.apply(lambda x: 0 if bool(female_pattern.search(str(x))) else (1 if bool(male_pattern.search(str(x))) else x))
    
    def _create_age_groups(self, column: pd.Series) -> pd.Series:
        """Creates age groups from the specified column."""
        column = pd.to_numeric(column, errors='coerce')
        column.dropna(inplace=True)
        
        bin_edges = [0] + [100 / self.num_groups * i for i in range(1, self.num_groups + 1)]
        labels = list(range(self.num_groups))
        
        return pd.cut(column, bins=bin_edges, labels=labels)
    
    def set_group(self, labels: List[int]):
        """Sets the group column in the dataset."""
        self.labels['group'] = labels
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single item from the dataset at the given index."""
        image_path = f"{self.image_data_dir}/{self.labels.loc[idx, self.image_column]}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transforms()(image)
        else:
            image = self.initial_transform(image)
        
        label = torch.tensor([self.labels.loc[idx, 'labels']]).int()
        gender = self.labels.loc[idx, 'gender_group'].int()
        age = self.labels.loc[idx, 'age_group'].int()
        group = self.labels.loc[idx, 'group'].int()
        
        return {
            'image': image,
            'label': label,
            'gender': gender,
            'age': age,
            'group': group
        }

class DataModule(pl.LightningDataModule):
    """Data module for handling dataset operations in a PyTorch Lightning pipeline."""

    def __init__(self, dataset: Type[TorchDataset], data_dir: str, image_data_dir: str, task: str, transform: bool,
                 batch_size: int = 32, fraction: float = 1, num_workers: int = 11, num_groups: int = 4):
        """Initializes the data module."""
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.image_data_dir = image_data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.fraction = fraction
        self.num_workers = num_workers
        self.task = task
        self.num_groups = num_groups
        self.save_hyperparameters()

        cutmix = transforms.CutMix(num_classes=1)
        mixup = transforms.MixUp(num_classes=1) 
        self.cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Any:
        """Collates a batch using either CutMix or MixUp."""
        return self.cutmix_or_mixup(*default_collate(batch))

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up the dataset for training, validation, or testing."""
        if stage == "fit":
            self.dataset_train = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                              type='train', transform=self.transform, fraction=self.fraction, 
                                              task=self.task, num_groups=self.num_groups)
            self.dataset_val = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                            type='val', transform=False, task=self.task, num_groups=self.num_groups)

        if stage == "test":
            self.dataset_test = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                             type='test', transform=False, task=self.task, num_groups=self.num_groups)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset."""
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset."""
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset."""
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)