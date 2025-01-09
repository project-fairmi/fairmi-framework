from typing import Type
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd

from torch.utils.data import default_collate

class Dataset(torch.utils.data.Dataset):
    """Base dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str, transform: bool = False, fraction: float = 1):
        """constructor.

        Args:
            data_dir (str): path to the dataset
            type (str): type of the dataset (train, val, test)
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use
            task (str): Task to perform
        """
        random.seed(42)
        self.image_data_dir = image_data_dir
        self.type = type
        self.transform = transform
        self.data_dir = data_dir
        self.fraction = fraction

        self.initial_transform = transforms.Compose([
            transforms.ToImage(),  
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    def transforms(self):
        return transforms.Compose([
            self.initial_transform,
            transforms.RandAugment(num_ops=4)
        ])
    
    def convert_gender_to_binary(self, column:pd.Series, female_value:any = "Female", male_value:any = "Male") -> pd.Series:
        """Converts gender values in a pandas Series to binary format.

        Args:
            column (pd.Series): A pandas Series containing gender values.
            female_value (Any): The value representing female in the column. Default is "Female".
            male_value (Any): The value representing male in the column. Default is "Male".

        Returns:
            pd.Series: A pandas Series with binary gender values (0 for female, 1 for male).
        """
        return column.map({female_value: 0, male_value: 1})
    
    def create_age_groups(self, column: pd.Series, num_groups: int = 4) -> pd.Series:
        """Creates age groups from the specified column and returns the new column.

        Args:
            column (pd.Series): A pandas Series containing patient ages.
            num_groups (int): Number of age groups. Default is 4.

        Returns:
            pd.Series: A pandas Series containing the age groups.
        """
        column = pd.to_numeric(column, errors='coerce')
        column.dropna(inplace=True)
        
        bin_edges = [0] + [100 / num_groups * i for i in range(1, num_groups + 1)]
        
        labels = list(range(num_groups))
        
        age_group = pd.cut(column, bins=bin_edges, labels=labels)
        return age_group.astype(int)

    def create_groups(self):
        #TODO: create groups using hierarchical clustering kmeans on the features of the dataset
        ...
        
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: Type[Dataset], data_dir: str, image_data_dir: str, task: str, transform: bool,
                  batch_size: int = 32, fraction: float = 1, num_workers: int  = 11, num_groups: int = 4):
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
        return self.cutmix_or_mixup(*default_collate(batch))

    def setup(self, stage: str) -> None:
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
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)