from ..base import Dataset, DataModule

class Brax(Dataset):
    """BRAX, a Brazilian labeled chest X-ray dataset.

    The Brax dataset contains X-ray images for classification tasks, including
    Pneumonia detection. The dataset is available at:
    https://physionet.org/content/brax/1.1.0/

    Attributes:
        data_dir (str): Path to the directory containing the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        labels (pd.DataFrame): DataFrame containing metadata and labels.
    """

    def __init__(self,
                 data_dir: str,
                 image_data_dir: str,
                 type: str,
                 labels_file: str = 'master_spreadsheet_update.csv',
                 image_column: str = 'PngPath',
                 transform: bool = False,
                 fraction: float = 1,
                 task: str = 'Pneumonia',
                 num_groups: int = 4,
                 patient_id_column: str = 'PatientID',
                 age_column: str = 'PatientAge',
                 gender_column: str = 'PatientSex') -> None:
        """Initializes the Brax dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (e.g., 'train', 'val', 'test').
            labels_file (str): Name of the file containing the labels.
            image_column (str): Name of the column containing image IDs.
            transform (bool): Whether to apply transformations to the images.
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

def BraxModule(batch_size: int = 32,  
               num_workers: int = 4,
               **kwargs) -> DataModule:
    """Creates a DataModule for the Brax dataset.

    Args:
        batch_size (int): Size of each batch of data.
        num_workers (int): Number of worker threads for data loading.
        **kwargs: Additional keyword arguments for initializing the dataset.

    Returns:
        DataModule: A DataModule instance configured for the Brax dataset.
    """
    return DataModule(
        dataset=Brax, 
        batch_size=batch_size,  
        num_workers=num_workers,
        **kwargs
    )