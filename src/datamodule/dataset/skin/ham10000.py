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
                 task: str = 'mel',
                 num_groups: int = 4,
                 patient_id_column: str = 'lesion_id',
                 age_column: str = 'age',
                 gender_column: str = 'sex',
                 **kwargs) -> None:
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
        """Configures the dataset for the HAM10000 dataset.

        This method augments the labels DataFrame by creating dummy variables
        for the diagnosis column ('dx') and then calls the superclass configuration.
        It also replaces any zero values in the age column with one to work with the
        age_groups correct in the dataset, if use 0 results in NaN groups.
        It adds a binary column 'malignant' where benign lesions are 0 and malignant are 1.
        """
        
        # Create dictionary mapping diagnosis types to binary classification
        malignant_map = {
            'nv': 0,    # Melanocytic nevi (benign)
            'bkl': 0,   # Benign keratosis-like lesions (benign)
            'df': 0,    # Dermatofibroma (benign)
            'vasc': 0,  # Vascular lesions (benign)
            'mel': 1,   # Melanoma (malignant)
            'bcc': 1,   # Basal cell carcinoma (malignant)
            'akiec': 1  # Actinic keratoses and intraepithelial carcinoma (malignant)
        }
        
        # Create binary classification column
        self.labels['malignant'] = self.labels['dx'].map(malignant_map)
        
        # Continue with existing code
        dummies = pd.get_dummies(self.labels['dx'])
        self.labels = pd.concat([self.labels, dummies], axis=1)
        self.labels[self.age_column] = self.labels[self.age_column].replace(0, 1)
        self.labels = self.labels[self.labels[self.gender_column].isin(['male', 'female'])]
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