from src.datamodule.dataset.vision.brset import Brset
from src.config import config
from base import BaseDatasetTest


class TestBrset(BaseDatasetTest):
    
    @classmethod
    def create(cls, **kwargs):
        """Create a Brset dataset instance with the given parameters."""
        
        default_params = {
            'data_dir': config['data']['brset']['data_path'],
            'image_data_dir': config['data']['brset']['image_data_path'],
            'type': 'train',
            'labels_file': 'labels.csv',
            'image_column': 'image_id',
            'transform': False,
            'fraction': 1,
            'task': 'diabetic_retinopathy',
            'num_groups': 2,
            'patient_id_column': 'patient_id',
            'age_column': 'patient_age',
            'gender_column': 'patient_sex',
        }

        default_params.update(kwargs)

        return Brset(**default_params)
