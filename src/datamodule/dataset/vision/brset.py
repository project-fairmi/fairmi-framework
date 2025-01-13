import torch
import pandas as pd
import random
from PIL import Image
from ..dataset import Dataset, DataModule

class Brset(Dataset):
    """Brset dataset."""

    def __init__(self, data_dir: str, image_data_dir: str, type: str,
                transform: bool = False, fraction: float = 1, task: str = 'diabetic_retinopathy', 
                num_groups: int = 4, patient_id_column: str = 'patient_id', 
                age_column: str = 'patient_age', gender_column: str = 'patient_sex'):
        """Brset dataset constructor.

        Args:
            data_dir (str): path to the dataset
            image_data_dir (str): path to the directory containing image data
            type (str): type of the dataset (train, val, test)
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use
            task (str): Task to perform
            num_groups (int): Number of groups for stratification
            patient_id_column (str): Name of the column containing patient IDs
            age_column (str): Name of the column containing patient ages
            gender_column (str): Name of the column containing patient gender
        """
        super().__init__(data_dir, image_data_dir, type, transform, fraction, age_column, gender_column, num_groups, task, patient_id_column)
        random.seed(42)
        self.image_data_dir = image_data_dir
        self.transform = transform
        self.data_dir = data_dir

        self.labels = pd.read_csv(f"{self.data_dir}/labels_brset.csv")

        self.configure_dataset()
        self.split()

    def configure_dataset(self):
        """Configure the dataset."""
        super().configure_dataset()

        positive_cases = self.labels[self.labels[self.task] == 1]
        negative_cases = self.labels[self.labels[self.task] == 0]

        n_positive = len(positive_cases)
        sampled_negative_cases = negative_cases.sample(n=n_positive)

        self.labels = pd.concat([positive_cases, sampled_negative_cases]).reset_index(drop=True)
        
    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be returned.

        Returns:
            Tuple: A tuple containing:
                - image (PIL.Image): The image at the specified index.
                - label (torch.Tensor): The label associated with the image.
                - gender (str): The gender of the patient.
                - age_group (str): The age group of the patient.
        """
        # image_path = f"{self.image_data_dir}/{self.labels.loc[idx, 'image_id']}.jpg"
        # image = Image.open(image_path).convert("RGB")
        image = Image.new('RGB', (224, 224))

        if self.transform:
            compose = self.transforms()
            image = compose(image)
        else:
            image = self.initial_transform(image)
        
        label = self.labels.loc[idx, 'labels']
        label = torch.tensor([label]).float()
        gender = self.labels.loc[idx, 'gender_group']
        age = self.labels.loc[idx, 'age_group']
        
        return {
            'image': image,
            'label': label,
            'gender': gender,
            'age': age
        }

def BrsetModule(data_dir: str, image_data_dir: str, transform: bool, task: str, batch_size: int = 32,
                fraction: float = 1, num_workers: int  = 11, num_groups: int = 4):
    return DataModule(dataset=Brset, data_dir=data_dir, image_data_dir=image_data_dir, transform=transform,
                            batch_size=batch_size, fraction=fraction, num_workers=num_workers, task=task, num_groups=num_groups)