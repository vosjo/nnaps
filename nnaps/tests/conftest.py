
import os
import pytest
from pathlib import Path

@pytest.fixture
def root_dir():
    """
    We need to use the os package when providing file paths for loading saved models in the FCpredictor and XGB
    predictor. For unknown reasons these don't work with the Path library.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def base_path():
    """
    Use the Path library to define the file path to the test folder. To be used when loading or saving test data.
    This works for almost all tests except for saving models in FCpredictor and XGBpredictor. For those use root_dir
    """
    return Path(__file__).parent