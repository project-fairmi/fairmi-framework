from typing import Union, List
from ..base import Dataset, DataModule

class CheXpert(Dataset):
    """CheXpert dataset for chest X-ray classification.

    The CheXpert dataset is a large dataset of chest X-rays for multi-label
    classification of 14 common chest radiographic observations. The dataset is 
    available at: https://stanfordmlgroup.github.io/competitions/chexpert/.

    Attributes:
        data_dir (str): Path to the directory containing the dataset.
        image_data_dir (str): Path to the directory containing image data.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
        patient_id_column (str): Name of the column containing patient IDs.
        path_column (str): Name of the column containing image paths.
    """

    def __init__(self,
                 data_dir: str,
                 type: str,
                 labels_file: Union[str, List[str]] = ['train.csv', 'valid.csv'],
                 image_column: str = None,
                 fraction: float = 1,
                 task: str = 'Pneumonia',
                 num_groups: int = 3,
                 patient_id_column: str = 'Patient',
                 age_column: str = 'Age',
                 gender_column: str = 'Sex',
                 path_column: str = 'Path',
                 image_data_dir: str = None,
                 **kwargs) -> None:
        """Initializes the CheXpert dataset.

        Args:
            data_dir (str): Path to the dataset.
            type (str): Type of the dataset (e.g., 'train', 'val', 'test').
            labels_file (Union[str, List[str]]): Name or list of names of the file(s) containing the labels.
            image_column (str): Name of the column containing image paths.
            transform (bool): Whether to apply transformations to the images.
            fraction (float): Fraction of the dataset to use.
            task (str): Task to perform (default: 'Pneumonia').
            num_groups (int): Number of groups for stratification.
            patient_id_column (str): Name of the column containing patient IDs.
            age_column (str): Name of the column containing patient ages.
            gender_column (str): Name of the column containing patient gender.
            path_column (str): Name of the column containing image paths.
            image_data_dir (str): Path to the directory containing image data.
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
            path_column=path_column,
            **kwargs
        )

        self.configure_dataset()
        self.split()

    def configure_dataset(self) -> None:
        """Configures the CheXpert dataset.

        This method preprocesses the labels DataFrame by:
          - Extracting the patient ID from the study path.
          - Updating the path to remove the '/CheXpert-v1.0-small/' segment.
          - Handling uncertain labels by converting -1 to 1 (per CheXpert paper recommendation).
        """
        # Handle uncertain labels: convert -1 to 1 if the task column exists.
        if self.task in self.labels.columns:
            self.labels[self.task] = self.labels[self.task].fillna(0).replace(-1, 1)
            
        super().configure_dataset()

        # Preprocess the labels file to extract patient ID from study path.
        self.labels[self.patient_id_column] = self.labels[self.path_column].apply(
            lambda x: x.split('/')[2]
        )

        # Update the path to remove '/CheXpert-v1.0-small/'.
        self.labels[self.path_column] = (
            self.labels[self.path_column]
            .apply(lambda x: f"{self.data_dir}/{x}")
            .apply(lambda x: x.replace('/CheXpert-v1.0-small/', '/'))
        )

        # Remove rows where gender is 'Unknown'
        self.labels = self.labels[self.labels[self.gender_column] != 'Unknown']

    def split(self) -> None:
        """Splits the dataset into training, validation, and test sets.

        This method assigns samples to 'train' or 'valid' based on the presence
        of 'train' or 'valid' in the path column.
        """
        if self.type == 'train':
            self.labels = self.labels[self.labels[self.path_column].str.contains('train', case=False)]
            if self.fraction < 1.0:
                self.labels = self.labels.groupby('labels').apply(
                    lambda x: x.sample(frac=self.fraction, random_state=42)
                ).reset_index(drop=True)
        elif self.type == 'val' or self.type in ['test', 'eval']:
            self.labels = self.labels[self.labels[self.path_column].str.contains('valid', case=False)]


def CheXpertModule(batch_size: int = 32,  
                   num_workers: int = 4,
                   **kwargs) -> DataModule:
    """Creates a DataModule for the CheXpert dataset.

    Args:
        batch_size (int): Batch size for data loading (default: 32).
        num_workers (int): Number of workers for data loading (default: 4).
        **kwargs: Additional keyword arguments for initializing the dataset,
                  such as data_dir, image_data_dir, transform, task, etc.

    Returns:
        DataModule: A DataModule instance configured for the CheXpert dataset.
    """
    return DataModule(
        dataset=CheXpert,
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )