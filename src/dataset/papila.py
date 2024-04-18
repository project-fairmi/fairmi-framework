import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import pandas as pd
import random
from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Papila(torch.utils.data.Dataset):
    """Papila dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str, transform: bool = False, fraction: float = 1):
        """Papila dataset constructor.

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

        self._labels = pd.read_excel(f"{self.data_dir}/patient_data_od.xlsx")
        self.configure_dataset()

    def configure_dataset(self):
        self._labels_od = pd.read_excel(f"{self.data_dir}/patient_data_od.xlsx")
        self._labels_od['Eye'] = 'OD'

        self._labels_os = pd.read_excel(f"{self.data_dir}/patient_data_os.xlsx")
        self._labels_os['Eye'] = 'OS'

        self._labels_od.drop(self._labels_od.index[0], inplace=True)
        self._labels_os.drop(self._labels_os.index[0], inplace=True)

        self._labels = pd.concat([self._labels_od, self._labels_os]).sort_index(kind='merge')

        self._labels.dropna(subset=self._labels.columns.difference(['Eye']), how='all', inplace=True)
        self._labels.reset_index(drop=True, inplace=True)
        self._labels['Diagnosis'] = self._labels['Diagnosis'].map({'no': 0, 'yes': 1, 'maybe': 1})

        numbers = list(range(self._labels["ID"].min(), self._labels["ID"].max() + 1))
        random.shuffle(numbers)
        
        train_size = int(0.6 * len(numbers))
        validation_size = int(0.1 * len(numbers))
        
        train_numbers = numbers[:train_size]

        remaining_numbers = numbers[train_size:]

        validation_numbers = remaining_numbers[:validation_size]
        test_numbers = remaining_numbers[validation_size:]

        if type == 'train':
            self.labels = self._labels[self._labels['ID'].isin(train_numbers)].reset_index()
        elif type == 'val':
            self.labels = self._labels[self._labels['ID'].isin(validation_numbers)].reset_index()
        elif type == 'test' or type == 'eval':
            self.labels = self._labels[self._labels['ID'].isin(test_numbers)].reset_index()
        else:
            raise ValueError(f'Invalid type: {type} (must be train, val or test/eval)')

    def transforms(self):
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
        image_path = f"{self.image_data_dir}/{self.labels.loc[idx, 'ID']}.jpg"
        image = Image.open(image_path).convert("RGB")
        
        image = self.initial_transform(image)
    
        if self.transform:
            compose = self.transforms()
            image = compose(image)
        
        label = self.labels.loc[idx, 'Diagnosis']
        
        return image, label


class PapilaModule(pl.LightningDataModule):
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
            self.papila_train = Papila(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='train', transform=self.transform, fraction=self.fraction)
            self.papila_val = Papila(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='val', transform=False)

        if stage == "test":
            self.papila_test = Papila(data_dir=self.data_dir, image_data_dir=self.image_data_dir, type='test', transform=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.papila_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.papila_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.papila_test, batch_size=self.batch_size)