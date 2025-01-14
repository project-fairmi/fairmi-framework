import torch
import pandas as pd
import random
from PIL import Image
from ..dataset import Dataset, DataModule

class Odir(Dataset):
    """ODIR dataset.

    Attributes:
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        data_dir (str): Path to the directory containing the dataset.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
    """

    def __init__(self, data_dir: str, image_data_dir: str, type: str, labels_file: str = 'full_df.csv',
                 image_column: str = 'filename', transform: bool = False, fraction: float = 1, task: str = 'D', 
                 num_groups: int = 4, patient_id_column: str = 'ID', 
                 age_column: str = 'Patient Age', gender_column: str = 'Patient Sex'):
        """Initializes the Odir dataset.

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

def OdirModule(data_dir: str, 
               image_data_dir: str, 
               transform: bool, 
               task: str, 
               batch_size: int = 32, 
               fraction: float = 1, 
               num_workers: int = 4,
               num_groups: int = 4) -> DataModule:
    """Creates a DataModule for the Odir dataset.

    Args:
        data_dir (str): Path to the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        task (str): Task to perform.
        batch_size (int): Batch size for data loading.
        fraction (float): Fraction of the dataset to use.
        num_workers (int): Number of workers for data loading.
        num_groups (int): Number of groups for stratification.

    Returns:
        DataModule: A DataModule instance configured for the Odir dataset.
    """
    return DataModule(dataset=Odir, 
                      data_dir=data_dir, 
                      image_data_dir=image_data_dir, 
                      transform=transform, 
                      batch_size=batch_size, 
                      fraction=fraction, 
                      num_workers=num_workers, 
                      task=task,
                      num_groups=num_groups)