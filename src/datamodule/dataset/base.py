from typing import List, Type, Optional, Dict, Union
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import v2 as transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import re
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from src.config import config

if config['data']['random_seed'] is not None:
    RANDOM_SEED = config['data']['random_seed']
else:
    RANDOM_SEED = 42

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

class Dataset(TorchDataset):
    """Base dataset for loading and processing data for machine learning tasks."""

    def __init__(self, data_dir: str, image_data_dir: str, labels_file: Union[str, List[str]], type: str, model_transform: transforms.Compose,
                 image_column: Optional[str] = None, augment: bool = False, fraction: float = 1, age_column: Optional[str] = None,
                 gender_column: Optional[str] = None, num_groups: int = 4, task: Optional[str] = None,
                 patient_id_column: Optional[str] = None, path_column: Optional[str] = None):
        """Initializes the dataset."""
        random.seed(RANDOM_SEED)
        self.image_data_dir = image_data_dir
        self.type = type
        self.model_transform = model_transform
        self.augment = augment
        self.data_dir = data_dir
        self.fraction = fraction
        self.age_column = age_column
        self.gender_column = gender_column
        self.num_groups = num_groups
        self.task = task
        self.patient_id_column = patient_id_column
        self.image_column = image_column
        self.path_column = path_column

        self.labels = self._set_labels(labels_file)
        self.transforms = self._get_transforms()

    def _set_labels(self, labels_file: str) -> pd.DataFrame:
        """Sets the labels for the dataset."""
        if isinstance(labels_file, list):
            return pd.concat([pd.read_csv(f"{self.data_dir}/{file}") for file in labels_file])
        elif labels_file.endswith('.csv'):
            return pd.read_csv(f"{self.data_dir}/{labels_file}")
        else:
            raise ValueError(f'Invalid file format: {labels_file} (must be .csv)')

    def _get_augmentation_transform(self) -> transforms.Compose:
        """Returns the augmentation transformation pipeline."""
        # Define augmentations here. Using RandAugment as before.
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))
            # transforms.RandAugment(num_ops=3)
        ])

    def _get_transforms(self) -> transforms.Compose:
        """Returns the final transformation pipeline, potentially combining model-specific and augmentation transforms."""
        final_transforms_list = []

        # Add model-specific transforms (or default if none provided)
        model_transform_to_add = self.model_transform
        if not model_transform_to_add:
            model_transform_to_add = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        
        # Flatten model transforms if it's a Compose object
        if isinstance(model_transform_to_add, transforms.Compose):
            final_transforms_list.extend(model_transform_to_add.transforms)
        else:
            # Handle cases where model_transform might be a single transform function
            final_transforms_list.append(model_transform_to_add)

        # Add augmentation transforms if enabled
        if self.augment and self.type == 'train':
            augmentation_transform = self._get_augmentation_transform()
            # Flatten augmentation transforms if it's a Compose object
            if isinstance(augmentation_transform, transforms.Compose):
                 final_transforms_list.extend(augmentation_transform.transforms)
            else:
                # Handle cases where augmentation_transform might be a single transform function
                final_transforms_list.append(augmentation_transform)

        return transforms.Compose(final_transforms_list)

    def configure_dataset(self):
        """Configures the dataset by creating age and gender groups, and preparing labels based on the task."""
        if self.age_column:
            self.labels = self.labels.dropna(subset=[self.age_column])
            self.labels = self.labels[self.labels[self.age_column] != 0]
            self.labels['age_group'] = self._create_age_groups(self.labels[self.age_column])
        else:
            self.labels['age_group'] = np.nan
        
        if self.gender_column:
            self.labels = self.labels.dropna(subset=[self.gender_column])
            self.labels['gender_group'] = self._convert_gender_to_binary(self.labels[self.gender_column])
            self.labels = self.labels[self.labels['gender_group'].isin([0, 1])]
        else:
            self.labels['gender_group'] = np.nan
        
        if 'group' not in self.labels.columns:
            self.labels['group'] = np.nan

        if self.task:
            self.labels[self.task] = self.labels[self.task].fillna(0)
            self.labels['labels'] = self.labels[self.task]
        
        if self.path_column is None and self.image_column is not None:
            self.labels['path'] = self.labels[self.image_column].apply(
                lambda x: f"{self.image_data_dir}/{x}" if str(x).lower().endswith('.jpg') else f"{self.image_data_dir}/{x}.jpg"
            )
            self.path_column = 'path'

    def split(self):
        """
            Splits the dataset into training, validation, and test sets based on patient IDs
            with proper stratification. Returns all three splits at once.
            
            Args:
                labels_df: DataFrame with metadata and labels
                patient_id_column: Column name for patient IDs
                label_column: Column name for the target labels
                
            Returns:
                train_df, val_df, test_df: The three splits as DataFrames
        """
        np.random.seed(RANDOM_SEED)

        patients = self.labels[self.patient_id_column].unique()
        np.random.shuffle(patients)

        patient_df = self.labels.groupby(self.patient_id_column)['labels'].agg(
             lambda x: x.value_counts().index[0]  # classe mais comum
        ).reset_index()

        train_patients, remaining_patients = train_test_split(
            patient_df[self.patient_id_column],
            test_size=(1 - TRAIN_SPLIT),
            random_state=RANDOM_SEED,
            stratify=patient_df['labels']
        )

        remaining_df = patient_df[patient_df[self.patient_id_column].isin(remaining_patients)]
        val_ratio = VAL_SPLIT / (1 - TRAIN_SPLIT)
            
        validation_patients, test_patients = train_test_split(
            remaining_df[self.patient_id_column],
            test_size=(1 - val_ratio),
            random_state=RANDOM_SEED,
            stratify=remaining_df['labels']
        )

        if self.type == 'train':    
            self.labels = self.labels[self.labels[self.patient_id_column].isin(train_patients)]
            if self.fraction < 1.0:
                # Amostragem estratificada para manter distribuição
                self.labels = self.labels.groupby('labels').apply(
                    lambda x: x.sample(frac=self.fraction, random_state=RANDOM_SEED)
                ).reset_index(drop=True)
            else:
                self.labels = self.labels.reset_index(drop=True)
        elif self.type == 'val':
            self.labels = self.labels[self.labels[self.patient_id_column].isin(validation_patients)].reset_index(drop=True)
        elif self.type in ['test', 'eval']:
            self.labels = self.labels[self.labels[self.patient_id_column].isin(test_patients)].reset_index(drop=True)
        else:
            raise ValueError(f'Invalid type: {self.type} (must be train, val or test/eval)')

    def _convert_gender_to_binary(self, column: pd.Series) -> pd.Series:
        """Converts gender values to binary format (0 for female, 1 for male)."""
        female_pattern = re.compile(r'^(f|feminine|female|woman|girl)$', re.IGNORECASE)
        male_pattern = re.compile(r'^(m|masculine|male|man|boy)$', re.IGNORECASE)

        return column.apply(lambda x: 0 if bool(female_pattern.search(str(x))) else (1 if bool(male_pattern.search(str(x))) else x))
    
    def _create_age_groups(self, column: pd.Series) -> pd.Series:
        """Creates age groups from the specified column."""
        column = pd.to_numeric(column, errors='coerce')
        
        bin_edges = [0] + [100 / self.num_groups * i for i in range(1, self.num_groups + 1)]
        labels = list(range(self.num_groups))
        
        return pd.cut(column, bins=bin_edges, labels=labels)
    
    def set_group(self, labels: List[int]):
        """Sets the group column in the dataset."""
        self.labels['group'] = labels

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        """Returns a single item from the dataset at the given index."""
        image_path = self.labels[self.path_column].iloc[idx]
        image = Image.open(image_path).convert("RGB")

        image = self.transforms(image)

        label = torch.tensor(self.labels.loc[idx, 'labels'], dtype=torch.long)
        gender = torch.tensor(self.labels.loc[idx, 'gender_group'], dtype=torch.long)
        age = torch.tensor(self.labels.loc[idx, 'age_group'], dtype=torch.long)
        group = self.labels.loc[idx, 'group']
        
        return {
            'image': image,
            'label': label,
            'gender': gender,
            'age': age,
            'group': group
        }

class DataModule(pl.LightningDataModule):
    """Data module for handling dataset operations in a PyTorch Lightning pipeline."""

    def __init__(self, dataset: Type[TorchDataset], data_dir: str, image_data_dir: str, task: str, model_transform: transforms.Compose,
                 augment_train: bool = False, batch_size: int = 32, fraction: float = 1, num_workers: int = 11, num_groups: int = 4):
        """Initializes the data module."""
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.image_data_dir = image_data_dir
        self.model_transform = model_transform # Now required
        self.augment_train = augment_train # Store the augmentation flag for training
        self.batch_size = batch_size
        self.fraction = fraction
        self.num_workers = num_workers
        self.task = task
        self.num_groups = num_groups
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up the dataset for training, validation, or testing."""
        if stage == "fit" or stage is None: # Ensure setup runs if stage is None (e.g., during testing without fit)
            self.dataset_train = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                              type='train', model_transform=self.model_transform, augment=self.augment_train,
                                              fraction=self.fraction, task=self.task, num_groups=self.num_groups)
            self.dataset_val = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                            type='val', model_transform=self.model_transform, augment=False, # No augmentation for validation
                                            task=self.task, num_groups=self.num_groups)

        if stage == "test" or stage is None:
             # Ensure setup runs if stage is None (e.g., during testing without fit)
             # Check if dataset_test already exists to avoid re-instantiation if setup(None) was called
            if not hasattr(self, 'dataset_test'):
                self.dataset_test = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                                type='test', model_transform=self.model_transform, augment=False, # No augmentation for test
                                                task=self.task, num_groups=self.num_groups)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset."""
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset."""
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset."""
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
