# Dataset Testing Framework

## Overview

The dataset testing framework defines a comprehensive set of tests to validate the behavior and data quality in dataset implementations. The tests are implemented in the abstract class `BaseDatasetTest`, which can be extended to test specific datasets.

## Implemented Tests

### Dataset Initialization

**Method:** `test_dataset_initialization`

Verifies the basic initialization properties of the dataset:
- Dataset is not empty
- Labels are loaded as a pandas DataFrame
- Fraction is within valid range (0,1]
- Dataset type is valid (train/val/test)

### Item Structure

**Method:** `test_dataset_item`

Verifies the structure and types of individual dataset items:
- Item contains all required fields (image, label, gender, age, group)
- Image is a PyTorch tensor
- Label is a PyTorch tensor
- Gender is a tensor of type int8 or contains NaN values
- Age is a tensor of type int8 or contains NaN values
- Group is a tensor of type int8 or contains NaN values

### NaN Verification

**Method:** `test_dataset_nan`

Verifies the absence of NaN values in dataset items:
- Checks the labels DataFrame
- Samples a subset of items (or all, if check_all=True)
- Checks for NaN in each field (image, label, gender, age)
- Fails on the first item with NaN found

### Number of Age Groups

**Method:** `test_num_age_groups`

Tests dataset behavior with different numbers of groups (1 to 10):
- Dataset is not empty
- Correct number of unique group values are present
- All group indices are within the valid range
- No NaN values in any dataset

### Dataset Fraction

**Method:** `test_fraction`

Tests dataset behavior with different fractions of the full dataset [0.1, 0.25, 0.5, 0.75, 1.0]:
- Correct dataset size for each fraction
- Dataset is not empty
- No NaN values in the fractioned dataset

### Image Consistency

**Method:** `test_random_image`

Tests the consistency of dataset items across different instances:
- Verifies that items retrieved by the same index from different dataset instances are identical
- Tests consistency in full and fractioned datasets
- Ensures deterministic data loading

## How to Use

To test a specific dataset, extend the `BaseDatasetTest` class and implement the `create()` method to instantiate the dataset to be tested:

```python
class MyDatasetTest(BaseDatasetTest):
    @classmethod
    def create(cls, **kwargs):
        # Configure defaults for your dataset if needed
        default_kwargs = {
            'transform': True,
            'fraction': 1.0,
            'type': 'train'
        }
        default_kwargs.update(kwargs)
        return MyDataset(**default_kwargs)
```

The tests can then be run using pytest:

```bash
pytest -xvs test_my_dataset.py
```