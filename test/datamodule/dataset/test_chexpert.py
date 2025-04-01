from src.datamodule.dataset.chest.chexpert import CheXpert
from src.config import config
from base import BaseDatasetTest


class TestCheXpert(BaseDatasetTest):
    
    @classmethod
    def create(cls, **kwargs):
        """Create a CheXpert dataset instance with the given parameters."""
        
        default_params = {
            'data_dir': config['data']['chexpert']['data_path'],
            'image_data_dir': None,
            'type': 'train',
            'labels_file': ['train.csv', 'valid.csv'],
            'image_column': None,
            'transform': False,
            'fraction': 1,
            'task': 'Pneumonia',
            'num_groups': 3,
            'patient_id_column': 'Patient',
            'age_column': 'Age',
            'gender_column': 'Sex',
            'path_column': 'Path'
        }

        default_params.update(kwargs)

        return CheXpert(**default_params)