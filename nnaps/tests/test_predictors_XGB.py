import os
import pytest

import pandas as pd
import numpy as np

from nnaps import predictors

from pathlib import Path

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_path = Path(__file__).parent

@pytest.fixture(scope='module')
def testpredictor():
    setup = {
        'datafile': '/home/joris/Python/nnaps/nnaps/tests/BesanconGalactic_summary.txt',
        'default_encoder': 'OrdinalEncoder',
        'features': ['M1', 'Pinit'],
        'regressors': ['Pfinal', 'qfinal'],
        'classifiers': ['product']
    }

    predictor = predictors.XGBPredictor(setup=setup)

    predictor.fit()

    return predictor

class TestXGBPredictorSetup:

    def test_make_from_setup(self, testpredictor):

        for name in ['Pfinal', 'qfinal']:
            assert name in testpredictor.model
            assert testpredictor.model[name].__class__.__name__ == 'XGBRegressor'

        for name in ['product']:
            assert name in testpredictor.model
            assert testpredictor.model[name].__class__.__name__ == 'XGBClassifier'


class TestXGBPredictorTrainingPredicting:

    def test_train_model(self, testpredictor):

        res = testpredictor.predict(testpredictor.test_data)

        assert 'Pfinal' in res.columns
        assert 'qfinal' in res.columns
        assert 'product' in res.columns

class TestXGBPredictorSaveLoad:

    def test_save_load(self, testpredictor):

        try:
            testpredictor.save_model('XGB_test_model.dat')
            loadedpredictor = predictors.XGBPredictor(saved_model='XGB_test_model.dat')
        finally:
            os.remove('XGB_test_model.dat')

        d1 = testpredictor.predict(testpredictor.test_data)
        d2 = loadedpredictor.predict(loadedpredictor.test_data)

        pd.testing.assert_frame_equal(d1, d2)
