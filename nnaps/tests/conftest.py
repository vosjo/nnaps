
import os
import pytest

@pytest.fixture
def root_dir():
    """
    We need to use the os package when providing file paths for loading saved models in the FCpredictor and XGB
    predictor. For unknown reasons these don't work with the Path library.
    """
    return os.path.dirname(os.path.abspath(__file__))