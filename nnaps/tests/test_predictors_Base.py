import os
import pytest

import pandas as pd
import numpy as np

from nnaps import predictors

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# by using scope='module', the predictor is created once and then used for all functions
# that need it in this test module.
@pytest.fixture(scope='module')
def testpredictor():
    setup = {
        'datafile': '/home/joris/Python/nnaps/nnaps/tests/BesanconGalactic_summary.txt',
        'default_encoder': 'OrdinalEncoder',
        'features': ['M1', 'Pinit'],
        'regressors': ['Pfinal', 'qfinal'],
        'classifiers': ['product']
    }
    return predictors.XGBPredictor(setup=setup)

class TestBasePredictorTrainingPredicting:
    """
    These tests don't check the actual processing of the data, they only assure that
    the returned objects are of the correct type and shape
    """

    def test_process_features(self, testpredictor):

        X = testpredictor._process_features(testpredictor.test_data)

        assert type(X) == np.ndarray
        assert X.shape == (len(testpredictor.test_data), 2)

    def test_process_targets(self, testpredictor):

        # -- normal transform tests
        # ---------------------------
        data_df = testpredictor.test_data
        data_ar = []
        for col in testpredictor.regressors + testpredictor.classifiers:
            data_ar.append(data_df[col].values.reshape(-1, 1))

        Y = testpredictor._process_targets(data_df, inverse=False, return_df=False)

        assert type(Y) == list
        assert len(Y) == 3
        assert len(Y[0]) == len(testpredictor.test_data)

        Y = testpredictor._process_targets(data_df, inverse=False, return_df=True)

        assert type(Y) == pd.DataFrame
        assert Y.shape == (len(testpredictor.test_data), 3)
        for name in testpredictor.regressors + testpredictor.classifiers:
            assert name in Y.columns

        Y = testpredictor._process_targets(data_ar, inverse=False, return_df=False)

        assert type(Y) == list
        assert len(Y) == 3
        assert len(Y[0]) == len(testpredictor.test_data)

        data_ar_in = Y.copy()

        Y = testpredictor._process_targets(data_ar, inverse=False, return_df=True)

        assert type(Y) == pd.DataFrame
        assert Y.shape == (len(testpredictor.test_data), 3)
        for name in testpredictor.regressors + testpredictor.classifiers:
            assert name in Y.columns

        data_df_in = Y.copy()

        # -- inverse transform tests
        # ---------------------------

        Y = testpredictor._process_targets(data_df_in, inverse=True, return_df=False)

        assert type(Y) == list
        assert len(Y) == 3
        assert len(Y[0]) == len(testpredictor.test_data)

        Y = testpredictor._process_targets(data_df_in, inverse=True, return_df=True)

        assert type(Y) == pd.DataFrame
        assert Y.shape == (len(testpredictor.test_data), 3)
        for name in testpredictor.regressors + testpredictor.classifiers:
            assert name in Y.columns

        Y = testpredictor._process_targets(data_ar_in, inverse=True, return_df=False)

        assert type(Y) == list
        assert len(Y) == 3
        assert len(Y[0]) == len(testpredictor.test_data)

        Y = testpredictor._process_targets(data_ar_in, inverse=True, return_df=True)

        assert type(Y) == pd.DataFrame
        assert Y.shape == (len(testpredictor.test_data), 3)
        for name in testpredictor.regressors + testpredictor.classifiers:
            assert name in Y.columns

