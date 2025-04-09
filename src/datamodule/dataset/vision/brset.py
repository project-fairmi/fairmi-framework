import random
import pandas as pd
from ..base import Dataset, DataModule

class Brset(Dataset):
    """Brset dataset.

    This class represents the Brset dataset containing image data and associated metadata.

    Attributes:
        data_dir (str): Path to the directory containing the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Flag that determines whether transformations are applied.
        labels (pd.DataFrame): DataFrame containing metadata and labels.
    """

    def __init__(self,
                 data_dir: str,
                 image_data_dir: str,
                 type: str,
                 labels_file: str = 'labels.csv',
                 image_column: str = 'image_id',
                 fraction: float = 1,
                 task: str = 'diabetic_retinopathy',
                 num_groups: int = 4,
                 patient_id_column: str = 'patient_id',
                 age_column: str = 'patient_age',
                 gender_column: str = 'patient_sex',
                 **kwargs) -> None:
        """Initializes the Brset dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (train, val, test).
            labels_file (str): Name of the file containing the labels.
            image_column (str): Name of the column containing image IDs.
            transform (bool): Whether to apply transformations to samples.
            fraction (float): Fraction of the dataset to use.
            task (str): Task to perform.
            num_groups (int): Number of groups for stratification.
            patient_id_column (str): Name of the column containing patient IDs.
            age_column (str): Name of the column containing patient ages.
            gender_column (str): Name of the column containing patient genders.
        """
        random.seed(42)
        super().__init__(
            data_dir=data_dir,
            image_data_dir=image_data_dir,
            labels_file=labels_file,
            image_column=image_column,
            type=type,
            fraction=fraction,
            age_column=age_column,
            gender_column=gender_column,
            num_groups=num_groups,
            task=task,
            patient_id_column=patient_id_column,
            **kwargs
        )
        self.configure_dataset()
        self.split()

    def configure_dataset(self) -> None:
        """Configures the dataset by balancing positive and negative cases.

        This method separates positive and negative cases based on the task column,
        samples negative cases to match the number of positive cases, and concatenates
        them into a balanced DataFrame.
        """
        super().configure_dataset()
        positive_cases = self.labels[self.labels[self.task] == 1]
        negative_cases = self.labels[self.labels[self.task] == 0]
        n_positive = len(positive_cases)
        sampled_negative_cases = negative_cases.sample(n=n_positive, random_state=42)
        self.labels = pd.concat([positive_cases, sampled_negative_cases]).reset_index(drop=True)

def BrsetModule(batch_size: int = 32,
                num_workers: int = 1,
                **kwargs) -> DataModule:
    """Creates a DataModule for the Brset dataset.

    Args:
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        **kwargs: Additional keyword arguments including:
            - data_dir (str): Path to the dataset.
            - image_data_dir (str): Path to the directory containing image data.
            - transform (bool): Whether to apply transformations to the images.
            - task (str): Task to perform.
            - fraction (float): Fraction of the dataset to use.
            - num_groups (int): Number of groups for stratification.

    Returns:
        DataModule: A DataModule instance configured for the Brset dataset.
    """
    return DataModule(
        dataset=Brset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )