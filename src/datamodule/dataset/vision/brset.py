import torch
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from ..dataset import Dataset, DataModule
from imblearn.over_sampling import RandomOverSampler
import numpy as np


class Brset(Dataset):
    """Brset dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str,
                transform: bool = False, fraction: float = 1, balance: str = 'equal',
                task: str = 'diabetic_retinopathy', num_groups: int = 4):
        """Brset dataset constructor.

        Args:
            data_dir (str): path to the dataset
            type (str): type of the dataset (train, val, test)
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use
            balance (str): Balance the dataset: equal, over_sampling
            task (str): Task to perform
        """
        super().__init__(data_dir, image_data_dir, type, transform, fraction)
        random.seed(42)
        self.image_data_dir = image_data_dir
        self.type = type
        self.transform = transform
        self.data_dir = data_dir
        self.fraction = fraction
        self.balance = balance
        self.task = task
        self.num_groups = num_groups
        
        self._labels = pd.read_csv(f"{self.data_dir}/labels.csv")
        self.configure_dataset()
    
    @staticmethod
    def clean_data(df: pd.DataFrame = None):
        """Clean the dataset.

        Args:
            df (pd.DataFrame): dataset to be cleaned
        
        Returns:
            pd.DataFrame: cleaned dataset
        """
        df.dropna(subset=['patient_age'], inplace=True)
        df['patient_sex'] = df['patient_sex'] - 1
        return df

    @classmethod
    def pos_weight(cls, data_dir: str, task: str = 'diabetic_retinopathy'):
        """Return the positive weight of the dataset.

        Args:
            data_dir (str): path to the dataset
        
        Returns:
            float: positive weight of the dataset
        """
        df = pd.read_csv(f"{data_dir}/labels.csv")
        df = cls.clean_data(df)
        return len(df[df[task] == 0]) / len(df[df[task] == 1])


    def configure_dataset(self):
        """Configure the dataset."""
        self._labels = Brset.clean_data(self._labels)
        self._labels['age_group'] = self.create_age_groups(self._labels['patient_age'], num_groups=self.num_groups)
        random_seed = 42
        np.random.seed(random_seed)

        if self.balance == 'equal':
            positive_cases = self._labels[self._labels[self.task] == 1]
            negative_cases = self._labels[self._labels[self.task] == 0]

            n_positive = len(positive_cases)
            sampled_negative_cases = negative_cases.sample(n=n_positive)

            balanced_data = pd.concat([positive_cases, sampled_negative_cases]).reset_index(drop=True)
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

            n_total = len(balanced_data)
            n_train = int(n_total * 0.6)
            n_val = int(n_total * 0.1)

            train_data = balanced_data[:n_train]
            validation_data = balanced_data[n_train:n_train + n_val]
            test_data = balanced_data[n_train + n_val:]
        if self.balance == 'over_sampling' and self.type == 'train':
            # Splitting data by patient_id
            unique_patient_ids = self._labels['patient_id'].unique()
            train_ids, test_val_ids = train_test_split(unique_patient_ids, test_size=0.4)

            train_data = self._labels[self._labels['patient_id'].isin(train_ids)]
            test_val_data = self._labels[self._labels['patient_id'].isin(test_val_ids)]

            validation_data, test_data = train_test_split(test_val_data, test_size=2/3)

            train_data = train_data.reset_index(drop=True)
            validation_data = validation_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            # Apply oversampling
            ros = RandomOverSampler()
            X_train = train_data.drop(columns=[self.task])
            y_train = train_data[self.task]
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

            # Combine resampled data back into a DataFrame
            train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
            train_data[self.task] = y_resampled

        if self.type == 'train':
            self.labels = train_data.reset_index(drop=True)
            self.labels = self.labels.sample(frac=self.fraction).reset_index(drop=True) 
        elif self.type == 'val':
            self.labels = validation_data.reset_index(drop=True)
        elif self.type == 'test' or self.type == 'eval':
            self.labels = test_data.reset_index(drop=True)
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
        image_path = f"{self.image_data_dir}/{self.labels.loc[idx, 'image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
    
        if self.transform:
            compose = self.transforms()
            image = compose(image)
        else:
            image = self.initial_transform(image)
        
        label = self.labels.loc[idx, self.task]
        label = torch.tensor([label]).float()
        gender = self.labels.loc[idx, 'patient_sex']
        age_group = self.labels.loc[idx, 'age_group']

        return image, label, gender, age_group
    

def BrsetModule(data_dir: str, image_data_dir: str, transform: bool, task: str, batch_size: int = 32,
                fraction: float = 1, num_workers: int  = 11, num_groups: int = 4):
    return DataModule(dataset=Brset, data_dir=data_dir, image_data_dir=image_data_dir, transform=transform,
                            batch_size=batch_size, fraction=fraction, num_workers=num_workers, task=task, num_groups=num_groups)