from ..base import Dataset, DataModule
class CheXpert(Dataset):
    """CheXpert dataset for chest X-ray classification.

    The CheXpert dataset is a large dataset of chest X-rays for multi-label classification
    of 14 common chest radiographic observations. The dataset is available at:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    Attributes:
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        data_dir (str): Path to the directory containing the dataset.
        labels (pd.DataFrame): DataFrame containing the metadata and labels.
    """

    def __init__(self, data_dir: str, type: str, labels_file: str = ['train.csv', 'valid.csv'],
                 image_column: str = None, transform: bool = False, fraction: float = 1, task: str = 'Pneumonia', 
                 num_groups: int = 3, patient_id_column: str = 'Patient', 
                 age_column: str = 'Age', gender_column: str = 'Sex', path_column: str = 'Path', image_data_dir: str = None):
        """Initializes the CheXpert dataset.

        Args:
            data_dir (str): Path to the dataset.
            image_data_dir (str): Path to the directory containing image data.
            type (str): Type of the dataset (train, val, test).
            labels_file (str): Name of the file containing the labels.
            image_column (str): Name of the column containing image paths.
            transform (bool): Optional transform to be applied on a sample.
            fraction (float): Fraction of the dataset to use.
            task (str): Task to perform (default: 'Pneumonia').
            num_groups (int): Number of groups for stratification.
            patient_id_column (str): Name of the column containing patient IDs.
            age_column (str): Name of the column containing patient ages.
            gender_column (str): Name of the column containing patient gender.
        """
        super().__init__(data_dir=data_dir, image_data_dir=image_data_dir, labels_file=labels_file, 
                image_column=image_column, type=type, transform=transform, fraction=fraction, 
                age_column=age_column, gender_column=gender_column, num_groups=num_groups, 
                task=task, patient_id_column=patient_id_column, path_column=path_column)
        
        self.configure_dataset()
        self.split()

    def configure_dataset(self):
        super().configure_dataset()

        # Preprocess the labels file to extract patient ID from study path
        self.labels[self.patient_id_column] = self.labels[self.path_column].apply(lambda x: x.split('/')[2])

        # remove the /CheXpert-v1.0-small/ from the path
        self.labels[self.path_column] = self.labels[self.path_column].apply(lambda x: f"{self.data_dir}/{x}").apply(lambda x: x.replace('/CheXpert-v1.0-small/', '/'))
        
        # Handle uncertain labels (convert -1 to 1 as per CheXpert paper recommendation)
        if self.task in self.labels.columns:
            self.labels[self.task] = self.labels[self.task].fillna(0).replace(-1, 1)

def CheXpertModule(data_dir: str, 
                   image_data_dir: str, 
                   transform: bool, 
                   task: str, 
                   batch_size: int = 32, 
                   fraction: float = 1, 
                   num_workers: int = 4,
                   num_groups: int = 4) -> DataModule:
    """Creates a DataModule for the CheXpert dataset.

    Args:
        data_dir (str): Path to the dataset.
        image_data_dir (str): Path to the directory containing image data.
        transform (bool): Whether to apply transformations to the images.
        task (str): Task to perform (one of: 'Atelectasis', 'Cardiomegaly', 'Consolidation', 
                   'Edema', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', etc.).
        batch_size (int): Batch size for data loading (default: 32).
        fraction (float): Fraction of the dataset to use (default: 1).
        num_workers (int): Number of workers for data loading (default: 4).
        num_groups (int): Number of groups for stratification (default: 4).

    Returns:
        DataModule: A DataModule instance configured for the CheXpert dataset.
    """
    return DataModule(dataset=CheXpert, 
                      data_dir=data_dir, 
                      image_data_dir=image_data_dir, 
                      transform=transform, 
                      batch_size=batch_size, 
                      fraction=fraction, 
                      num_workers=num_workers, 
                      task=task,
                      num_groups=num_groups)