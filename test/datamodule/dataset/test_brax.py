from src.datamodule.dataset.chest.brax import Brax
from src.config import config
from base import BaseDatasetTest


class TestBrax(BaseDatasetTest):
    
    @classmethod
    def create(cls, **kwargs):
        """Create a Brax dataset instance with the given parameters."""
        
        default_params = {
            'data_dir': config['data']['brax']['data_path'],
            'image_data_dir': config['data']['brax']['image_data_path'],
            'type': 'train',
            'labels_file': 'master_spreadsheet_update.csv',
            'image_column': 'PngPath',
            'transform': False,
            'fraction': 1,
            'task': 'Pneumonia',
            'num_groups': 4,
            'patient_id_column': 'PatientID',
            'age_column': 'PatientAge',
            'gender_column': 'PatientSex',
        }

        default_params.update(kwargs)

        return Brax(**default_params)