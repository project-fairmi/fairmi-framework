import torch
import pandas as pd
import random
from PIL import Image
from ..dataset import Dataset, DataModule

class Brax(Dataset):
    """BRAX, a Brazilian labeled chest X-ray dataset.

    The Brax dataset contains X-ray images for classification tasks, including
    Pneumonia detection. The dataset is available at:
    https://physionet.org/content/brax/1.1.0/

    Attributes:
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        data_dir (str): Path to the directory containing the dataset.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
    """

    def __init__(self, data_dir: str, image_data_dir: str, type: str, labels_file: str = 'master_spreadsheet_update.csv',
                 image_column: str = 'PngPath', transform: bool = False, fraction: float = 1, task: str = 'Pneumonia', 
                 num_groups: int = 4, patient_id_column: str = 'PatientID', 
                 age_column: str = 'PatientAge', gender_column: str = 'PatientSex'):
        """Initializes the Brax dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (train, val, test).
            labels_file (str): Name of the file containing the labels.
            image_column (str): Name of the column containing image IDs.
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use.
            task (str): Task to perform.
            num_groups (int): Number of groups for stratification.
            patient_id_column (str): Name of the column containing patient IDs.
            age_column (str): Name of the column containing patient ages.
            gender_column (str): Name of the column containing patient gender.
        """
        super().__init__(data_dir, image_data_dir, labels_file, image_column, type, transform, fraction, age_column, gender_column, num_groups, task, patient_id_column)
        random.seed(42)

        self.configure_dataset()
        self.split()

def BraxModule(data_dir: str, 
               image_data_dir: str, 
               transform: bool, 
               task: str, 
               batch_size: int = 32, 
               fraction: float = 1, 
               num_workers: int = 4,
               num_groups: int = 4) -> DataModule:
    """Creates a DataModule for the Brax dataset.

    Args:
        data_dir (str): Path to the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        task (str): Task to perform (default: 'Pneumonia').
        batch_size (int): Batch size for data loading (default: 32).
        fraction (float): Fraction of the dataset to use (default: 1).
        num_workers (int): Number of workers for data loading (default: 4).
        num_groups (int): Number of groups for stratification (default: 4).

    Returns:
        DataModule: A DataModule instance configured for the Brax dataset.
    """
    return DataModule(dataset=Brax, 
                      data_dir=data_dir, 
                      image_data_dir=image_data_dir, 
                      transform=transform, 
                      batch_size=batch_size, 
                      fraction=fraction, 
                      num_workers=num_workers, 
                      task=task,
                      num_groups=num_groups)