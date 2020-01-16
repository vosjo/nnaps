import os
import pytest
import numpy as np

from nnaps import predictors


@pytest.fixture
def cleanup_files():
    yield
    if os.path.isfile('integration_training_history.csv'):
        os.remove('integration_training_history.csv')
    if os.path.isfile('integration_test_model.h5'):
        os.remove('integration_test_model.h5')

class TestIntegration:

    def test_integration(self, cleanup_files):

        predictor = predictors.BPS_predictor(setup_file='test_setup.yaml')

        predictor.fit(epochs=100, batch_size=128)

        y_test_1 = predictor.predict(predictor.test_data)

        predictor.save_model('integration_test_model.h5')

        assert os.path.isfile('integration_test_model.h5')

        predictor.save_training_history('integration_training_history.csv')

        assert os.path.isfile('integration_training_history.csv')

        predictor = predictors.BPS_predictor(saved_model='integration_test_model.h5')

        y_test_2 = predictor.predict(predictor.test_data)

        for par in predictor.regressors:
            np.testing.assert_almost_equal(y_test_1[par].values, y_test_2[par].values)
        for par in predictor.classifiers:
            np.testing.assert_array_equal(y_test_1[par].values, y_test_2[par].values)

