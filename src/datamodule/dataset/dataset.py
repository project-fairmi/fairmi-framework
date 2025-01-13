from typing import Type
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import re
import numpy as np
from torch.utils.data import default_collate

class Dataset(torch.utils.data.Dataset):
    """Base dataset for loading and processing data for machine learning tasks.

    This dataset handles preprocessing of data including transformations, splitting data into 
    training, validation, and test sets, and converting categorical data (e.g., gender) to numeric values.

    Attributes:
        image_data_dir (str): Path to the directory containing image data.
        type (str): Type of the dataset (train, val, test).
        transform (bool): Whether to apply transformations to the images.
        data_dir (str): Path to the directory containing the dataset.
        fraction (float): Fraction of the dataset to use.
        age_column (str): Name of the column containing the age information.
        gender_column (str): Name of the column containing gender information.
        num_groups (int): Number of age groups for stratification.
        task (str): Task to perform (e.g., 'Pneumonia' for classification).
        patient_id_column (str): Name of the column containing patient IDs.
    """

    def __init__(self, data_dir: str, image_data_dir: str, type: str, transform: bool = False,
                 fraction: float = 1, age_column: str = None, gender_column: str = None, num_groups: int = 4,
                 task: str = None, patient_id_column: str = None):
        """Initializes the dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of dataset (train, val, test).
            transform (bool): Whether to apply transformations to the images.
            fraction (float): Fraction of the dataset to use.
            age_column (str): Name of the column containing the age information.
            gender_column (str): Name of the column containing gender information.
            num_groups (int): Number of age groups for stratification.
            task (str): Task to perform (e.g., 'Pneumonia').
            patient_id_column (str): Name of the column containing patient IDs.
        """
        random.seed(42)
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

        self.initial_transform = transforms.Compose([
            transforms.ToImage(),  
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    def configure_dataset(self):
        """Configures the dataset by creating age and gender groups, and preparing labels based on the task.

        This method will process the gender and age data columns, creating new columns for age groups
        and binary gender values, and it will adjust the labels according to the task specified (e.g., classification).
        """
        if self.age_column is not None:
            self.labels = self.labels.dropna(subset=[self.age_column])
            self.labels['age_group'] = self.create_age_groups(self.labels[self.age_column], num_groups=self.num_groups)
        
        if self.gender_column is not None:
            self.labels = self.labels.dropna(subset=[self.gender_column])
            self.labels['gender_group'] = self.convert_gender_to_binary(self.labels[self.gender_column])
        
        if self.task is not None:
            self.labels[self.task] = self.labels[self.task].fillna(0)
            self.labels['labels'] = self.labels[self.task]

    def split(self):
        """Splits the dataset into training, validation, and test sets based on patient IDs.

        This method will shuffle the patient IDs and split them into training (60%), validation (10%), 
        and test (30%) sets based on the 'type' of dataset requested (train, val, test/eval).
        """
        random_seed = 42
        np.random.seed(random_seed)

        self.patients = self.labels[self.patient_id_column].unique()
        random.shuffle(self.patients)

        train_size = int(0.6 * len(self.patients))
        validation_size = int(0.1 * len(self.patients))
        
        train_numbers = self.patients[:train_size]
        remaining_numbers = self.patients[train_size:]

        validation_numbers = remaining_numbers[:validation_size]
        test_numbers = remaining_numbers[validation_size:]

        if self.type == 'train':    
            self.labels = self.labels[self.labels[self.patient_id_column].isin(train_numbers)].reset_index()
            self.labels = self.labels.sample(frac=self.fraction).reset_index(drop=True)
        elif self.type == 'val':
            self.labels = self.labels[self.labels[self.patient_id_column].isin(validation_numbers)].reset_index()
        elif self.type == 'test' or self.type == 'eval':
            self.labels = self.labels[self.labels[self.patient_id_column].isin(test_numbers)].reset_index()
        else:
            raise ValueError(f'Invalid type: {self.type} (must be train, val or test/eval)')

    def transforms(self):
        """Returns the transformation pipeline for the dataset.

        Returns:
            transforms.Compose: A composed transformation function.
        """
        return transforms.Compose([
            self.initial_transform,
            transforms.RandAugment(num_ops=4)
        ])

    def convert_gender_to_binary(self, column: pd.Series) -> pd.Series:
        """Converts gender values to binary format (0 for female, 1 for male).

        Args:
            column (pd.Series): A pandas Series containing gender values.

        Returns:
            pd.Series: A pandas Series with binary gender values (0 for female, 1 for male).
        """
        female_pattern = re.compile(r'^(f|feminine|female|woman|girl)$', re.IGNORECASE)
        male_pattern = re.compile(r'^(m|masculine|male|man|boy)$', re.IGNORECASE)

        return column.apply(lambda x: 0 if bool(female_pattern.search(str(x))) else (1 if bool(male_pattern.search(str(x))) else x))
    
    def create_age_groups(self, column: pd.Series, num_groups: int = 4) -> pd.Series:
        """Creates age groups from the specified column.

        Args:
            column (pd.Series): A pandas Series containing patient ages.
            num_groups (int): Number of age groups to create. Default is 4.

        Returns:
            pd.Series: A pandas Series containing the age groups.
        """
        column = pd.to_numeric(column, errors='coerce')
        column.dropna(inplace=True)
        
        bin_edges = [0] + [100 / num_groups * i for i in range(1, num_groups + 1)]
        
        labels = list(range(num_groups))
        
        age_group = pd.cut(column, bins=bin_edges, labels=labels)
        
        return age_group

    def create_groups(self):
        """Creates groups using hierarchical clustering or k-means on the dataset features.
        
        TODO: Implement hierarchical clustering or k-means to generate groups based on dataset features.
        """
        pass
        
class DataModule(pl.LightningDataModule):
    """Data module for handling dataset operations in a PyTorch Lightning pipeline.

    This class sets up the dataset, splits it into training, validation, and test sets, and provides
    data loaders for each set.

    Args:
        dataset (Type[Dataset]): The dataset class to use.
        data_dir (str): Path to the dataset.
        image_data_dir (str): Path to the directory containing image data.
        task (str): Task to perform.
        transform (bool): Whether to apply transformations to the data.
        batch_size (int): Batch size for data loading.
        fraction (float): Fraction of the dataset to use.
        num_workers (int): Number of workers for data loading.
        num_groups (int): Number of groups for stratification.
    """

    def __init__(self, dataset: Type[Dataset], data_dir: str, image_data_dir: str, task: str, transform: bool,
                 batch_size: int = 32, fraction: float = 1, num_workers: int  = 11, num_groups: int = 4):
        """Initializes the data module.

        Args:
            dataset (Type[Dataset]): The dataset class to use.
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            task (str): Task to perform.
            transform (bool): Whether to apply transformations to the data.
            batch_size (int): Batch size for data loading.
            fraction (float): Fraction of the dataset to use.
            num_workers (int): Number of workers for data loading.
            num_groups (int): Number of groups for stratification.
        """
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

    def collate_fn(self, batch):
        """Collates a batch using either CutMix or MixUp.

        Args:
            batch (list): A list of data samples.

        Returns:
            Collated batch.
        """
        return self.cutmix_or_mixup(*default_collate(batch))

    def setup(self, stage: str) -> None:
        """Sets up the dataset for training, validation, or testing.

        Args:
            stage (str): The stage of the pipeline ('fit', 'test').
        """
        if stage == "fit":
            self.dataset_train = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                               type='train', transform=self.transform, fraction=self.fraction, task=self.task,
                                                num_groups=self.num_groups)
            self.dataset_val = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                             type='val', transform=False, task=self.task,
                                             num_groups=self.num_groups)

        if stage == "test":
            self.dataset_test = self.dataset(data_dir=self.data_dir, image_data_dir=self.image_data_dir,
                                              type='test', transform=False, task=self.task,
                                              num_groups=self.num_groups)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training.
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation.
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for testing.
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)