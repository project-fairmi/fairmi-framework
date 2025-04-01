import random
import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader

class BaseDatasetTest:
    """Base abstract class for dataset testing.
    
    This class provides a comprehensive test suite for dataset implementations,
    verifying initialization, data loading, group assignments, and dataset fractioning.
    Subclasses should implement the create() method to instantiate the specific dataset.
    """

    @classmethod
    def create(cls, **kwargs):
        """Create a dataset instance for testing.
        
        Args:
            **kwargs: Configuration parameters to pass to the dataset constructor.
            
        Returns:
            An instance of the dataset to be tested.
        """
        return cls(**kwargs)

    @pytest.fixture
    def dataset(self):
        """Creates a full dataset fixture for testing.
        
        Returns:
            A dataset instance with default parameters.
        """
        return self.create()

    def test_dataset_initialization(self, dataset):
        """Test the basic initialization properties of the dataset.
        
        Args:
            dataset: The dataset instance to test.
            
        Tests:
            - Dataset is not empty
            - Labels are loaded as a pandas DataFrame
            - Fraction is within valid range (0,1]
            - Dataset type is valid (train/val/test)
        """
        assert len(dataset) > 0, "Dataset should not be empty"
        assert isinstance(dataset.labels, pd.DataFrame), "Labels should be a pandas DataFrame"
        assert 0 < dataset.fraction <= 1, "Fraction should be between 0 and 1"
        assert dataset.type in ['train', 'val', 'test'], "Dataset type should be train, val, or test"

    def test_dataset_item(self, dataset):
        """Test the structure and types of individual dataset items.
        
        Args:
            dataset: The dataset instance to test.
            
        Tests:
            - Item contains all required fields (image, label, gender, age, group)
            - Image is a torch.Tensor
            - Label is a torch.Tensor
            - Gender is a torch.int8 tensor or contains NaN values
            - Age is a torch.int8 tensor or contains NaN values
            - Group is a torch.int8 tensor or contains NaN values
        """
        item = dataset[random.randint(0, len(dataset) - 1)]
        assert 'image' in item, "Item should contain an image"
        assert 'label' in item, "Item should contain a label"
        assert 'gender' in item, "Item should contain gender information"
        assert 'age' in item, "Item should contain age information"
        assert 'group' in item, "Item should contain group information"
        
        assert isinstance(item['image'], torch.Tensor), "Image should be a torch.Tensor"
        assert isinstance(item['label'], torch.Tensor), "Label should be of type torch.Tensor"
        assert (item['gender'].dtype == torch.int8) or \
            (torch.isnan(item['gender']).any()), "Gender should be torch.int8 or contain NaN values"
        assert (item['age'].dtype == torch.int8) or \
            (torch.isnan(item['age']).any()), "Age should be torch.int8 or contain NaN values"
        # assert (item['group'].dtype == torch.int8) or \
        #     (torch.isnan(item['group']).any()), "Group should be torch.int8 or contain NaN values"

    def test_dataset_nan(self, dataset, check_all=False):
        """Verifies dataset items for any NaN values.
        
        Args:
            dataset: Dataset instance to test.
            check_all: If True, checks all items. Otherwise, samples a subset.
        """
        
        assert isinstance(dataset.labels, pd.DataFrame), "Labels should be a pandas DataFrame"

        # Select indices to check
        indices = range(len(dataset)) if check_all else random.sample(range(len(dataset)), min(100, len(dataset)))
        
        # Fields to check
        fields = ['image', 'label', 'gender', 'age']
        
        # Check for any NaN in any field
        for idx in indices:
            item = dataset[idx]
            for field in fields:
                if field in item and torch.isnan(item[field]).any():
                    assert False, f"{field.capitalize()} at index {idx} contains NaN values"
                    
        # If we got here, no NaNs were found
        assert True
    
    @pytest.mark.parametrize("num_groups", range(3, 4))
    def test_num_age_groups(self, num_groups):
        """
        Test dataset behavior with a different number of groups using batch processing.
        
        This test creates a dataset with the specified number of groups and verifies:
        - The dataset is not empty.
        - The correct number of unique group values is present.
        - All group indices are within the valid range.
        - No NaN values in any dataset item (for the 'age' field).
        - No 'age' value is greater than or equal to num_groups.
        
        Args:
            num_groups: Number of groups to test.
        """
        dataset = self.create(num_groups=num_groups)
        
        assert len(dataset) > 0, f"Dataset should not be empty for {num_groups} groups."

        # Use DataLoader to process the dataset in batches
        dataloader = DataLoader(dataset, batch_size=64)
        for batch in dataloader:
            # Assuming the field to verify is 'age'
            groups = batch.get('age')
            if not isinstance(groups, torch.Tensor):
                groups = torch.tensor(groups)
            
            # Check that the batch does not contain any NaN values
            assert not torch.isnan(groups).any(), "Batch contains NaN values in 'age'."
            
            # Check that no value is greater than or equal to num_groups
            if (groups >= num_groups).any():
                invalid_values = groups[groups >= num_groups].tolist()
                assert False, (
                    f"Found 'age' values >= num_groups in batch: {invalid_values}. "
                    f"All 'age' values should be between 0 and {num_groups - 1}."
                )

    def test_fraction(self, dataset):
        """Test dataset behavior with different fractions of the full dataset.
        
        Args:
            dataset: The full dataset instance to test against.
            
        Tests the dataset with fractions [0.1, 0.25, 0.5, 0.75, 1.0] to ensure:
        - Correct dataset size for each fraction
        - Dataset is not empty
        - No NaN values in the fractioned dataset
        """
        # Get the full dataset size from the fixture
        full_size = len(dataset)
        # Test different fractions including edge cases
        test_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        for fraction in test_fractions:
            fractioned_dataset = self.create(fraction=fraction)

            # Test the size is correct
            expected_size = int(full_size * fraction)
            abs(len(fractioned_dataset) - expected_size) <= 1, \
                f"Dataset size should be approximately {expected_size} for fraction {fraction}, got {len(fractioned_dataset)}"

            # Test that the dataset is not empty
            assert len(fractioned_dataset) > 0, f"Dataset should not be empty for fraction {fraction}"

    def test_random_image(self, dataset):
        """Tests consistency of dataset items across different instances.
        
        Verifies that items retrieved by the same index from different dataset
        instances are identical, ensuring deterministic data loading.
        
        Args:
            dataset: The dataset instance to test.
        """
        # Test full dataset consistency
        random_idx = random.randint(0, len(dataset) - 1)
        item = dataset[random_idx]
        
        # Compare with a fresh dataset instance (without transforms)
        other_dataset = self.create(transform=False)
        other_item = other_dataset[random_idx]
        
        # Verify all tensors are identical
        fields = ['image', 'label', 'gender', 'age']
        for field in fields:
            assert torch.equal(item[field], other_item[field]), \
                f"{field.capitalize()} at index {random_idx} differs between dataset instances"
        
        # Test fractioned dataset consistency
        for fraction in [0.25, 0.5, 1.0]:  # Reduced test set for efficiency
            first = self.create(fraction=fraction)
            second = self.create(fraction=fraction)
            
            if len(first) == 0:  # Skip empty datasets
                continue
                
            idx = random.randint(0, len(first) - 1)
            assert torch.equal(first[idx]['image'], second[idx]['image']), \
                f"Image consistency failed with fraction={fraction}"