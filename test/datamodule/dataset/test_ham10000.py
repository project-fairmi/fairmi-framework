from src.datamodule.dataset.skin.ham10000 import Ham10000
from src.config import config
from base import BaseDatasetTest

# filepath: /scratch/unifesp/fairmi/dilermando.queiroz/fairmi-framework/test/datamodule/dataset/test_ham10000.py


class TestHam10000(BaseDatasetTest):

    @classmethod
    def create(cls, **kwargs):
        """Create a HAM10000 dataset instance with the given parameters."""
        
        default_params = {
            'data_dir': config['data']['ham10000']['data_path'],
            'image_data_dir': config['data']['ham10000']['image_data_path'],
            'type': 'train',
            'labels_file': 'HAM10000_metadata.csv',
            'image_column': 'image_id',
            'augment': False,
            'fraction': 1,
            'task': 'mel',
            'num_groups': 4,
            'patient_id_column': 'lesion_id',
            'age_column': 'age',
            'gender_column': 'sex'
        }

        default_params.update(kwargs)
        return Ham10000(**default_params)