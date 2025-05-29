import argparse
import uuid
import importlib

import yaml
from src.model.classification import VisionTransformerModel
from src.config import config
from src.scaling_laws.utils.metric import Metric

import lightning as pl
import os
import pandas as pd

class TestModelExperiment:
    def __init__(self, versions: list[int], dataset: str ='brset', name: str ='vit-fairness'):
        """Initializes the TestModelExperiment class.

        Args:
            versions (list): A list of versions to test.
            dataset (str): The dataset to use for testing.
            name (str): The name of the experiment.
        """
        self.versions = versions
        self.name = name
        self.model_files = self._find_model_files()
        self.model_data_percentage = self._find_model_data('fraction')
        self.model_data_pretain = self._find_model_data('pretrain')
        self.model_num_groups = self._find_model_data('num_groups')[0]
        self.dataset = dataset
        self.datamodule = self._load_datamodule(dataset)
        self.id = str(uuid.uuid4())
        self.path = f"{config['log']['experiment_dir']}/{self.id}"
        os.makedirs(self.path, exist_ok=True)

    def _version_dir(self, version: int):
        return os.path.join(f"{config['log']['log_dir']}/{self.name}", f'version_{version}')

    def _find_model_data(self, information):
        """Finds the percentage of data used to train each model in a hparams.yaml.
        """

        model_data_percentage = {}
        for version in self.versions:
            version_dir = self._version_dir(version)
            with open(os.path.join(version_dir, 'hparams.yaml')) as f:
                hparams = yaml.load(f, Loader=yaml.FullLoader)
            model_data_percentage[version] = hparams[information]

        return model_data_percentage

    def _find_model_files(self):
        """Finds all .ckpt files in the specified version folder.
        """
        model_files = {}
        for version in self.versions:
            version_dir = os.path.join(f"{config['log']['log_dir']}/{self.name}", f'version_{version}/checkpoints')
            files = [f'{version_dir}/{f}' for f in os.listdir(version_dir) if f.endswith('.ckpt')]
            model_files[version] = files

        return model_files

    def _load_datamodule(self, name):
        """Dynamically loads the appropriate dataset module based on the dataset name.

        Args:
            name (str): The name of the dataset to load.

        Returns:
            DataModule: The loaded dataset module.
        """
        # Get dataset configuration from config
        dataset_config = config['data'][name]

        # Construct the module path and class name
        module_path = f"src.datamodule.dataset.vision.{name.capitalize()}Module"
        class_name = f"{name.capitalize()}Module"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the module class
            module_class = getattr(module, class_name)

            # Create the datamodule instance with parameters from config
            return module_class(
                data_dir=dataset_config['data_path'],
                image_data_dir=dataset_config['image_data_path'],
                transform=False,
                task=dataset_config['task'],
                num_groups=self.model_num_groups
            )
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load dataset module for '{name}': {e}")

    def load_model(self, model_path):
        """Loads a PyTorch model from a .ckpt file.

        Args:
            model_path (str): Path to the .ckpt file.
        """
        model = VisionTransformerModel.load_from_checkpoint(model_path)
        return model

    def test_model(self, model):
        """Tests the model using PyTorch Lightning Trainer and returns metrics.

        Args:
            model (pl.LightningModule): The model to test.
        """
        trainer = pl.Trainer(devices=1)
        return trainer.test(model, datamodule=self.datamodule)

    def run_tests(self):
        """Runs tests for all models and saves the results to a CSV file.
        """
        results = []
        for version, files in self.model_files.items():
            for file in files:
                model = self.load_model(file)
                metrics = self.test_model(model)
                fraction = self.model_data_percentage[version]
                pretrain = self.model_data_pretain[version]
                results.append({
                    'dataset': self.dataset,
                    'version': version,
                    'model': file,
                    'fraction': fraction,
                    'pretrain': pretrain,
                    **metrics[0]
                })

        self.results_df = pd.DataFrame(results)
        self.save_results()

    def save_results(self):
        """
        Saves the results to a CSV file.
        """
        self.results_df.to_csv(f"{self.path}/results.csv", index=False)

if __name__ == '__main__':    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model experiments.')
    parser.add_argument('--experiments', nargs='+', type=int, required=True, help='List of experiment values')
    parser.add_argument('--name', type=str, default='default-experiment', help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='brset', help='Dataset to use for testing')
    args = parser.parse_args()

    # Create and run the experiment
    experiment = TestModelExperiment(args.experiments, dataset=args.dataset, name=args.name)
    experiment.run_tests()
    metric_calculator = Metric(experiment.results_df, experiment.model_num_groups, experiment.path)
    metric_calculator.run_metrics()
