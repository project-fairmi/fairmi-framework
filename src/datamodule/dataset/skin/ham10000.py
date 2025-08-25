import pandas as pd
from ..base import Dataset, DataModule
from sklearn.model_selection import train_test_split
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
                 fraction: float = 1,
                 task: str = 'mel',
                 num_groups: int = 4,
                 patient_id_column: str = 'lesion_id',
                 age_column: str = 'age',
                 gender_column: str = 'sex',
                 random_seed: int = 42,
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
        self.random_seed = random_seed
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

        # Create dictionary specifically for melanoma detection
        melanoma_map = {
            'nv': 0,    # Melanocytic nevi (not melanoma)
            'bkl': 0,   # Benign keratosis-like lesions (not melanoma)
            'df': 0,    # Dermatofibroma (not melanoma)
            'vasc': 0,  # Vascular lesions (not melanoma)
            'mel': 1,   # Melanoma (positive)
            'bcc': 0,   # Basal cell carcinoma (not melanoma)
            'akiec': 0  # Actinic keratoses (not melanoma)
        }
        
        label_mapping = {
            'akiec': 0,
            'bcc': 1, 
            'bkl': 2,
            'df': 3,
            'mel': 4,
            'nv': 5,
            'vasc': 6
        }

        # Create a 7 class classification column
        self.labels['multi'] = pd.Categorical(self.labels['dx'])
        self.labels['multi'] = self.labels['multi'].map(label_mapping)

        # Create binary classification column
        self.labels['malignant'] = self.labels['dx'].map(malignant_map)

        # Create melanoma detection column
        self.labels['melanoma'] = self.labels['dx'].map(melanoma_map)
        
        # Continue with existing code
        self.labels[self.age_column] = self.labels[self.age_column].replace(0, 1)
        self.labels = self.labels[self.labels[self.gender_column].isin(['male', 'female'])]
        super().configure_dataset()

    def split(self):
        # Identify lesions with only one image (unique lesions)
        lesion_counts = self.labels.groupby(self.patient_id_column)['image_id'].count()
        unique_lesions = set(lesion_counts[lesion_counts == 1].index)

        # Mark whether each row corresponds to a unique lesion
        self.labels['is_unique'] = self.labels[self.patient_id_column].isin(unique_lesions)

        # Split unique lesions into train and test sets (70% train, 30% test)
        unique_data = self.labels[self.labels['is_unique']]
        
        # First split: separate train (70%) from test portion (30%)
        train_data, test_portion = train_test_split(
            unique_data, 
            test_size=0.3, 
            random_state=101, 
            stratify=unique_data['labels']
        )
        
        # Second split: divide test portion into validation and test
        # Split the 30% test portion in: 10% validation + 20% test
        val_data, test_data = train_test_split(
            test_portion, 
            test_size=0.66,
            random_state=101, 
            stratify=test_portion['labels']
        )

        # Create image ID sets for efficient lookup
        val_image_ids = set(val_data['image_id'])
        test_image_ids = set(test_data['image_id'])

        # Assign final dataset based on type
        if self.type == 'train':
            # Keep only training images (exclude validation and test images)
            excluded_ids = val_image_ids.union(test_image_ids)
            self.labels = self.labels[~self.labels['image_id'].isin(excluded_ids)].reset_index(drop=True)
            
            if self.fraction < 1.0:
                # Amostragem estratificada para manter distribuição
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