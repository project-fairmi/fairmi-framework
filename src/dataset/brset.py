import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Brset(torch.utils.data.Dataset):
    """Brset dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str, transform: bool = False, fraction: float = 1):
        """Brset dataset constructor.

        Args:
            data_dir (str): path to the dataset
            type (str): type of the dataset (train, val, test)
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use
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
        
        self._labels = pd.read_csv(f"{self.data_dir}/labels.csv")
        self.configure_dataset()
    
    def configure_dataset(self):
        """Configure the dataset.
        
        Drop rows with missing values, convert patient
        Change patient sex for 1, 2 to 0, 1, respectively
        Create age groups based on patient age
        Create a balanced dataset for diabetic retinopathy
        Split the dataset into train, validation and test sets
        """
        self._labels.dropna(subset=['patient_age'], inplace=True)
        self._labels['patient_sex'] = self._labels['patient_sex'] - 1
        self._labels['age_group'] = pd.cut(self._labels['patient_age'], bins=[0, 25, 50, 75, 100], labels=[0, 1, 2, 3])
        self._labels['age_group'] = self._labels['age_group'].astype(int)

        positive_cases = self._labels[self._labels['diabetic_retinopathy'] == 1]
        negative_cases = self._labels[self._labels['diabetic_retinopathy'] == 0]

        n_positive = len(positive_cases)
        sampled_negative_cases = negative_cases.sample(n=n_positive, random_state=42)

        self.balanced_data = pd.concat([positive_cases, sampled_negative_cases]).reset_index(drop=True)

        unique_patient_ids = self.balanced_data['patient_id'].unique()
        train_ids, test_val_ids = train_test_split(unique_patient_ids, test_size=0.4, random_state=42)

        train_data = self.balanced_data[self.balanced_data['patient_id'].isin(train_ids)]
        test_val_data = self.balanced_data[self.balanced_data['patient_id'].isin(test_val_ids)]

        validation_data, test_data = train_test_split(test_val_data, test_size=2/3, random_state=42)

        train_data = train_data.reset_index(drop=True)
        validation_data = validation_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        
        if self.type == 'train':
            self.labels = train_data.reset_index(drop=True)
            self.labels = train_data.sample(frac=self.fraction, random_state=42).reset_index(drop=True)
        elif self.type == 'val':
            self.labels = validation_data.reset_index(drop=True)
        elif self.type == 'test' or self.type == 'eval':
            self.labels = test_data.reset_index(drop=True)
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test/eval)')

    def transforms(self):
        """Return the transforms to be applied on the dataset.
        """
        return transforms.Compose([
            transforms.ToImage(),  
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomErasing(),
        ])
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Return the item at index idx.

        Args:
            idx (int): index of the item to be returned
        """
        image_path = f"{self.image_data_dir}/{self.labels.loc[idx, 'image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
    
        if self.transform:
            compose = self.transforms()
            image = compose(image)
        else:
            image = self.initial_transform(image)
        
        label = self.labels.loc[idx, 'diabetic_retinopathy']

        if self.type == 'test' or self.type == 'eval':
            gender = self.labels.loc[idx, 'patient_sex']
            age_group = self.labels.loc[idx, 'age_group']

            return image, label, gender, age_group
        
        return image, label
    
class BrsetModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, image_data_dir: str, transform: bool, batch_size: int = 32, fraction: float = 1):
        super().__init__()
        self.data_dir = data_dir
        self.image_data_dir = image_data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.fraction = fraction

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.brset_train = Brset(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='train', transform=self.transform, fraction=self.fraction)
            self.brset_val = Brset(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='val', transform=False)

        if stage == "test":
            self.brset_test = Brset(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='test', transform=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.brset_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.brset_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.brset_test, batch_size=self.batch_size)