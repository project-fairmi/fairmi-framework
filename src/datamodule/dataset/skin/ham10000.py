import pandas as pd
from ..base import Dataset, DataModule

class Ham10000(Dataset):
    """HAM10000 dataset.

    This class represents the HAM10000 dataset containing image data
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
                 labels_file: str = 'HAM10000_metadata.csv',
                 image_column: str = 'image_id',
                 transform: bool = False,
                 fraction: float = 1,
                 task: str = 'diagnosis',
                 num_groups: int = 4,
                 patient_id_column: str = 'lesion_id',
                 age_column: str = 'age',
                 gender_column: str = 'sex') -> None:
        """Initializes the HAM10000 dataset.

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
            gender_column (str): Name of the column containing patient gender.
        """
        super().__init__(
            data_dir=data_dir,
            image_data_dir=image_data_dir,
            labels_file=labels_file,
            image_column=image_column,
            type=type,
            transform=transform,
            fraction=fraction,
            age_column=age_column,
            gender_column=gender_column,
            num_groups=num_groups,
            task=task,
            patient_id_column=patient_id_column
        )
        self.configure_dataset()
        self.split()

    def configure_dataset(self) -> None:
        """Configures the dataset for the HAM10000 dataset.

        This method augments the labels DataFrame by creating dummy variables
        for the diagnosis column ('dx') and then calls the superclass configuration.
        """
        dummies = pd.get_dummies(self.labels['dx'])
        self.labels = pd.concat([self.labels, dummies], axis=1)
        super().configure_dataset()

def Ham10000Module(batch_size: int = 32,
                   num_workers: int = 4,
                   **kwargs) -> DataModule:
    """Creates a DataModule for the HAM10000 dataset.

    Args:
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        **kwargs: Additional keyword arguments for initializing the dataset,
                  such as data_dir, image_data_dir, transform, task, etc.

    Returns:
        DataModule: A DataModule instance configured for the HAM10000 dataset.
    """
    return DataModule(
        dataset=Ham10000,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )