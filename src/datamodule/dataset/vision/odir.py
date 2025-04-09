from ..base import Dataset, DataModule

class Odir(Dataset):
    """ODIR dataset.

    This class represents the ODIR dataset, containing image data and associated metadata.

    Attributes:
        data_dir (str): Path to the dataset directory.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Flag indicating whether to apply transformations to the images.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
    """

    def __init__(self,
                 data_dir: str,
                 image_data_dir: str,
                 type: str,
                 labels_file: str = 'full_df.csv',
                 image_column: str = 'filename',
                 fraction: float = 1,
                 task: str = 'D',
                 num_groups: int = 4,
                 patient_id_column: str = 'ID',
                 age_column: str = 'Patient Age',
                 gender_column: str = 'Patient Sex',
                 **kwargs) -> None:
        """Initializes the ODIR dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (e.g., 'train', 'val', 'test').
            labels_file (str, optional): File name containing the labels. Defaults to 'full_df.csv'.
            image_column (str, optional): Name of the column containing image IDs. Defaults to 'filename'.
            transform (bool, optional): Whether to apply transformations to the images. Defaults to False.
            fraction (float, optional): Fraction of the dataset to use. Defaults to 1.
            task (str, optional): Task to perform. Defaults to 'D'.
            num_groups (int, optional): Number of groups for stratification. Defaults to 4.
            patient_id_column (str, optional): Column name for patient IDs. Defaults to 'ID'.
            age_column (str, optional): Column name for patient ages. Defaults to 'Patient Age'.
            gender_column (str, optional): Column name for patient genders. Defaults to 'Patient Sex'.
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

def OdirModule(batch_size: int = 32,
               num_workers: int = 4,
               **kwargs) -> DataModule:
    """Creates a DataModule for the ODIR dataset.

    Args:
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        DataModule: A DataModule instance configured for the ODIR dataset.
    """
    return DataModule(
        dataset=Odir,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )