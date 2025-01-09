import torch
import pandas as pd
import random
from PIL import Image
from ..dataset import Dataset, DataModule
import numpy as np

class Odir(Dataset):
    """ODIR dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str,
                  transform: bool = False, fraction: float = 1, task: str = 'D', num_groups: int = 4):
        """ODIR dataset constructor.

        Args:
            data_dir (str): path to the dataset
            type (str): type of the dataset (train, val, test)
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use
        """
        super().__init__(data_dir, image_data_dir, type, transform, fraction)
        random.seed(42)
        self.image_data_dir = image_data_dir
        self.type = type
        self.transform = transform
        self.data_dir = data_dir
        self.fraction = fraction
        self.task = task
        self.num_groups = num_groups

        self.configure_dataset()

    def configure_dataset(self):
        """Configure the dataset.
        """
        random_seed = 42
        np.random.seed(random_seed)
        self.labels = pd.read_csv(f"{self.data_dir}/full_df.csv")
        self.labels['age_group'] = self.create_age_groups(self.labels['Patient Age'], num_groups=self.num_groups)
        self.labels['Patient Sex'] = self.convert_gender_to_binary(self.labels['Patient Sex'])

        self.labels['labels'] = self.labels['labels'].apply(lambda x: 1 if x == f"['{self.task}']" else 0)

        patients = self.labels["ID"].unique()
        random.shuffle(patients)
        
        train_size = int(0.6 * len(patients))
        validation_size = int(0.1 * len(patients))
        
        train_numbers = patients[:train_size]
        remaining_numbers = patients[train_size:]

        validation_numbers = remaining_numbers[:validation_size]
        test_numbers = remaining_numbers[validation_size:]

        if self.type == 'train':    
            self.labels = self.labels[self.labels['ID'].isin(train_numbers)].reset_index()
            self.labels = self.labels.sample(frac=self.fraction).reset_index(drop=True)
        elif self.type == 'val':
            self.labels = self.labels[self.labels['ID'].isin(validation_numbers)].reset_index()
        elif self.type == 'test' or self.type == 'eval':
            self.labels = self.labels[self.labels['ID'].isin(test_numbers)].reset_index()
        else:
            raise ValueError(f'Invalid type: {self.type} (must be train, val or test/eval)')

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Return the item at index idx.
        Args:
            idx (int): index of the item to be returned
        """
        image_path = f"{self.image_data_dir}/{self.labels.loc[idx, 'filename']}"
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            compose = self.transforms()
            image = compose(image)
        else:
            image = self.initial_transform(image)
        
        label = self.labels.loc[idx, 'labels']
        label = torch.tensor([label]).float()
        gender = self.labels.loc[idx, 'Patient Sex']
        age_group = self.labels.loc[idx, 'age_group']
        
        return image, label, gender, age_group

def OdirModule(data_dir: str, 
               image_data_dir: str, 
               transform: bool, 
               task: str, 
               batch_size: int = 32, 
               fraction: float = 1, 
               num_workers: int = 4,
               num_groups: int = 4):
    
    return DataModule(dataset=Odir, 
                             data_dir=data_dir, 
                             image_data_dir=image_data_dir, 
                             transform=transform, 
                             batch_size=batch_size, 
                             fraction=fraction, 
                             num_workers=num_workers, 
                             task=task,
                             num_groups=num_groups)