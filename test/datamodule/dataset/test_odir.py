from src.datamodule.dataset.vision.odir import Odir
from src.config import config
from base import BaseDatasetTest


class TestODIR(BaseDatasetTest):
    
    @classmethod
    def create(cls, **kwargs):
        """Create an ODIR dataset instance with the given parameters."""
        
        default_params = {
            'data_dir': config['data']['odir']['data_path'],
            'image_data_dir': config['data']['odir']['image_data_path'],
            'type': 'train',
            'labels_file': 'labels.csv',
            'image_column': 'image_id',
            'transform': False,
            'fraction': 1,
            'task': 'diabetic_retinopathy',
            'num_groups': 2,
            'patient_id_column': 'patient_id',
            'age_column': 'patient_age',
            'gender_column': 'patient_gender',
        }

        default_params.update(kwargs)

        return Odir(**default_params)