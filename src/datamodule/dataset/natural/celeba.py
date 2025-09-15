import pandas as pd
from ..base import Dataset, DataModule
from sklearn.model_selection import train_test_split
import numpy as np
from ..base import TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED

class CelebA(Dataset):
    """CelebA dataset.

    This class represents the CelebA dataset containing image data
    and associated metadata.

    Attributes:
        data_dir (str): Path to the directory containing the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
    """

    def __init__(self,
                 data_dir: str,
                 image_data_dir: str,
                 type: str,
                 labels_file: str = 'list_attr_celeba.csv',
                 partition_file: str = 'list_eval_partition.csv',
                 image_column: str = 'image_id',
                 fraction: float = 1,
                 task: str = 'Smiling',
                 num_groups: int = 2,
                 patient_id_column: str = 'image_id', # Using image_id as patient_id
                 age_column: str = 'Young', # No age column in CelebA attributes
                 gender_column: str = 'Male', # Using 'Male' attribute as gender
                 random_seed: int = 42,
                 **kwargs) -> None:
        """Initializes the CelebA dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (train, val, test).
            labels_file (str): Name of the file containing the attribute labels.
            partition_file (str): Name of the file containing the train/val/test partition.
            image_column (str): Name of the column containing image IDs.
            transform (bool): Whether to apply transformations to samples.
            fraction (float): Fraction of the dataset to use.
            task (str): Task to perform (e.g., 'Smiling', 'Attractive').
            num_groups (int): Number of groups for stratification.
            patient_id_column (str): Name of the column containing patient IDs (image_id for CelebA).
            age_column (str): Name of the column containing patient ages.
            gender_column (str): Name of the column containing patient gender ('Male' attribute for CelebA).
        """
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
        self.random_seed = random_seed
        self.partition_file = partition_file
        self.configure_dataset()
        self.split()

    def configure_dataset(self) -> None:
        """Configures the dataset for CelebA.

        This method loads the partition file, merges it with the attribute labels,
        converts attribute values from -1/1 to 0/1, and calls the superclass configuration.
        """
        partition_df = pd.read_csv(f"{self.data_dir}/{self.partition_file}")
        self.labels = pd.merge(self.labels, partition_df, on=self.image_column)

        # Convert -1/1 attributes to 0/1
        for col in self.labels.columns:
            if self.labels[col].isin([-1, 1]).all() and col != self.image_column:
                self.labels[col] = self.labels[col].apply(lambda x: 1 if x == 1 else 0)


        super().configure_dataset()

    def split(self):
        """Splits the dataset into training, validation, and test sets based on the partition file."""
        
        # Separate the test set (partition == 1)
        test_data = self.labels[self.labels['partition'] == 1].reset_index(drop=True)
        
        # Take the remaining data (partition == 0) for train and validation
        train_data = self.labels[self.labels['partition'] == 0].reset_index(drop=True)

        # Split train_val_data into training and validation
        test_data, val_data = train_test_split(
            test_data,
            test_size=0.1, # VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT), # Adjust test_size for the remaining split
            random_state=RANDOM_SEED,
            stratify=test_data['labels']
        )
        
        if self.type == 'train':
            self.labels = train_data.reset_index(drop=True)
            if self.fraction < 1.0:
                # Stratified sampling for training set
                self.labels = self.labels.groupby('labels').apply(
                    lambda x: x.sample(frac=self.fraction, random_state=self.random_seed)
                ).reset_index(drop=True)
            elif self.fraction > 1.0:
                self.labels = self.labels.groupby('labels').apply(
                    lambda x: x.sample(n=int(self.fraction), random_state=self.random_seed)
                ).reset_index(drop=True)
        elif self.type == 'val':
            self.labels = val_data.reset_index(drop=True)
        elif self.type in ['test', 'eval']:
            self.labels = test_data.reset_index(drop=True)
        else:
            raise ValueError(f'Invalid type: {self.type} (must be train, val or test/eval)')


def CelebAModule(batch_size: int = 32,
                   num_workers: int = 4,
                   **kwargs) -> DataModule:
    """Creates a DataModule for the CelebA dataset.

    Args:
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        **kwargs: Additional keyword arguments for initializing the dataset,
                  such as data_dir, image_data_dir, transform, task, etc.

    Returns:
        DataModule: A DataModule instance configured for the CelebA dataset.
    """
    return DataModule(
        dataset=CelebA,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
