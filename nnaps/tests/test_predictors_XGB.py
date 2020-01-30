import os
import pytest

import pandas as pd
import numpy as np

from nnaps import predictors

from pathlib import Path

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_path = Path(__file__).parent

class TestXGBPredictorSetup:

    def test_make_from_setup(self):

        setup = {
            'features': ['M1', 'Pinit'],
            'regressors': ['Pfinal', 'qfinal'],
            'classifiers': ['product']
        }

        predictor = predictors.XGBPredictor(setup=setup)

        for name in setup['regressors']:
            assert name in predictor.model
            assert predictor.model[name].__class__.__name__ == 'XGBRegressor'

        for name in setup['classifiers']:
            assert name in predictor.model
            assert predictor.model[name].__class__.__name__ == 'XGBClassifier'


class TestXGBPredictorTrainingPredicting:

    def test_train_model(self):
        setup = {
            'datafile': '/home/joris/Python/nnaps/nnaps/tests/BesanconGalactic_summary.txt',
            'default_encoder': 'OrdinalEncoder',
            'features': ['M1', 'Pinit'],
            'regressors': ['Pfinal', 'qfinal'],
            'classifiers': ['product']
        }

        predictor = predictors.XGBPredictor(setup=setup)

        predictor.fit()

        res = predictor.predict(predictor.test_data)

        assert 'Pfinal' in res.columns
        assert 'qfinal' in res.columns
        assert 'product' in res.columns
