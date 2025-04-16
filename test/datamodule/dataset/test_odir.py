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
            'labels_file': 'full_df.csv',
            'image_column': 'filename',
            'augment': False,
            'fraction': 1,
            'task': 'D',
            'num_groups': 2,
            'patient_id_column': 'ID',
            'age_column': 'Patient Age',
            'gender_column': 'Patient Sex',
            'model_transform': None
        }

        default_params.update(kwargs)

        return Odir(**default_params)