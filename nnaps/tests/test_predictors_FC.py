import os
import pytest

import pandas as pd
import numpy as np

import yaml

from sklearn import preprocessing

from nnaps import predictors

from pathlib import Path

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_path = Path(__file__).parent


class TestBPSPredictorSetup:

    def test_make_from_setup(self):

        predictor = predictors.FCPredictor(setup_file=base_path / 'test_setup.yaml')
        Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
        Yregressors_ = ['Pfinal', 'qfinal']
        Yclassifiers_ = ['product', 'binary_type']

        # test reading the X and Y variables
        assert all([a in predictor.features for a in Xpars_])
        assert all([a in predictor.regressors for a in Yregressors_])
        assert all([a in predictor.classifiers for a in Yclassifiers_])

        # test making the preprocessors
        pp = predictor.processors

        for par in Xpars_:
            assert par in pp, "{} does not have a preprocessor".format(par)
            assert pp[par].__class__ == preprocessing.StandardScaler, \
                            "{} does not have the correct preprocessor.".format(par) + \
                            " expected {}, got {}".format(preprocessing.StandardScaler, pp[par].__class__)

        for par in Yregressors_:
            assert par in pp, "{} does not have a preprocessor".format(par)
            assert pp[par] == None, "{} does not have the correct preprocessor.".format(par) + \
                                                " expected None, got {}".format(pp[par].__class__)

        for par in Yclassifiers_:
            assert par in pp, "{} does not have a preprocessor".format(par)
            assert pp[par].__class__ == preprocessing.OneHotEncoder, \
                            "{} does not have the correct preprocessor.".format(par) + \
                            " expected {}, got {}".format(preprocessing.OneHotEncoder,pp[par].__class__)

        # test making the model
        mod = predictor.model

        assert len(mod.layers) == 11, \
                         "Model does not have correct number of layers, expected" + \
                         " {}, got {}".format(11, len(mod.layers))

        assert mod.input.shape[1] == len(Xpars_), \
                         "Model input does not have correct shape, expected " + \
                         "{}, got {}".format(len(Xpars_),mod.input.shape[1])

        assert len(mod.output) == len(Yregressors_) + len(Yclassifiers_), \
                         "Model does not have correct number of outputs, expected" + \
                         " {}, got {}".format(len(Yregressors_) + len(Yclassifiers_), len(mod.output))

    def test_make_from_setup_with_data(self):
        setup = """
        features:
           - M1
           - qinit
        regressors:
           - Pfinal
        classifiers:
           - product
        """
        setup = yaml.safe_load(setup)

        data = pd.read_csv(base_path / 'BesanconGalactic_summary.txt')

        predictor = predictors.FCPredictor(setup=setup, data=data)

        assert predictor.train_data is not None
        assert predictor.test_data is not None

        assert len(predictor.train_data) + len(predictor.test_data) == len(data)

    def test_make_from_saved_model(self, root_dir):

        predictor = predictors.FCPredictor(saved_model=os.path.join(root_dir, 'test_model_FC.h5'))

        Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
        Yregressors_ = ['Pfinal', 'qfinal']
        Yclassifiers_ = ['product', 'binary_type']

        # test reading the X and Y variables
        assert all([a in predictor.features for a in Xpars_])
        assert all([a in predictor.regressors for a in Yregressors_])
        assert all([a in predictor.classifiers for a in Yclassifiers_])

        # test making the preprocessors
        pp = predictor.processors

        for par in Xpars_:
            assert par in pp, "{} does not have a preprocessor".format(par)
            assert pp[par].__class__ == preprocessing.MinMaxScaler

        for par in Yclassifiers_:
            assert par in pp, "{} does not have a preprocessor".format(par)
            assert pp[par].__class__ == preprocessing.OneHotEncoder

        # test making the model
        mod = predictor.model

        assert len(mod.layers) == 8, \
            "Model does not have correct number of layers, expected {}, got {}".format(8, len(mod.layers))

        assert mod.input.shape[1] == len(Xpars_), \
            "Model input does not have correct shape, expected {}, got {}".format(len(Xpars_), mod.input.shape[1])

        assert len(mod.output) == len(Yregressors_) + len(Yclassifiers_), \
            "Model does not have correct number of outputs, expected {}, got {}".format(len(Yregressors_) +
                                                                                        len(Yclassifiers_),
                                                                                        len(mod.output))

        # test that the training data is loaded again
        assert predictor.train_data is not None
        assert predictor.test_data is not None


class TestBPSPredictorTrainingPredicting:

    def test_process_features(self):

        data = np.random.normal(10, 3, size=100)
        df = pd.DataFrame(data={'M1':data, 'qinit':data})

        p = preprocessing.StandardScaler()
        p.fit(df[['M1']])

        processors = dict(M1=p,
                         qinit = p,)

        predictor = predictors.FCPredictor()

        predictor.processors = processors
        predictor.features = ['M1', 'qinit']

        X = predictor._process_features(df)

        # check that the output shape is correct
        assert X.shape == (100,2)

        # check that all arrays are transformed
        np.testing.assert_almost_equal(np.mean(X[:,0]), 0)
        np.testing.assert_almost_equal(np.std(X[:,0]), 1)
        np.testing.assert_almost_equal(np.mean(X[:,1]), 0)
        np.testing.assert_almost_equal(np.std(X[:,0]), 1)

    def test_process_targets(self):
        data = np.random.normal(10, 3, size=100)
        df = pd.DataFrame(data={'M1final': data, 'qfinal': data})

        p = preprocessing.StandardScaler()
        p.fit(df[['M1final']])

        processors = dict(M1final=p,
                          qfinal = None,)

        predictor = predictors.FCPredictor()

        predictor.processors = processors
        predictor.regressors = ['M1final', 'qfinal']

        Y = predictor._process_targets(df)

        # check that the output is a list
        assert type(Y) == list

        # check that both parameters are included
        assert len(Y) == 2

        # check that one array is transformed and the other not
        np.testing.assert_almost_equal(np.mean(Y[0]), 0)
        np.testing.assert_almost_equal(np.std(Y[0]), 1)
        np.testing.assert_array_equal(Y[1], df[['qfinal']].values)

    def test_append_to_history(self):

        predictor = predictors.FCPredictor()

        predictor.regressors = ['M1final', 'qfinal']
        predictor.classifiers = []

        data = {'M1final_mae': [0.3, 0.2],
                'val_M1final_mae': [0.31, 0.21],
                'M1final_loss': [1.5, 1.3],
                'val_M1final_loss': [1.6, 1.4],
                'qfinal_mae': [0.3, 0.2],
                'val_qfinal_mae': [0.31, 0.21],
                'qfinal_loss': [1.5, 1.3],
                'val_qfinal_loss': [1.6, 1.4],
                'training_run': [1, 1]}
        history1 = pd.DataFrame(data=data)
        history1.index.name = 'epoch'

        predictor.history = history1

        data = {'M1final_mae': [0.1, 0.0],
                'val_M1final_mae': [0.11, 0.01],
                'M1final_loss': [1.2, 1.1],
                'val_M1final_loss': [1.3, 1.2],
                'qfinal_mae': [0.1, 0.0],
                'val_qfinal_mae': [0.11, 0.01],
                'qfinal_loss': [1.2, 1.1],
                'val_qfinal_loss': [1.3, 1.2],
                }
        history2 = pd.DataFrame(data=data)
        history2.index.name = 'epoch'

        data = {'M1final_mae': [0.3, 0.2, 0.1, 0.0],
                'val_M1final_mae': [0.31, 0.21, 0.11, 0.01],
                'M1final_loss': [1.5, 1.3, 1.2, 1.1],
                'val_M1final_loss': [1.6, 1.4, 1.3, 1.2],
                'qfinal_mae': [0.3, 0.2, 0.1, 0.0],
                'val_qfinal_mae': [0.31, 0.21, 0.11, 0.01],
                'qfinal_loss': [1.5, 1.3, 1.2, 1.1],
                'val_qfinal_loss': [1.6, 1.4, 1.3, 1.2],
                'training_run': [1, 1, 2, 2]}
        history_expected = pd.DataFrame(data=data)
        history_expected.index.name = 'epoch'

        predictor._append_to_history(history2)

        history = predictor.history

        assert history.equals(history_expected), \
            "\nExpected dataframe: \n{}\nGot dataframe: \n {}".format(history_expected.to_string(),
                                                                      history.to_string())

    def test_train_model(self):

        # WARNING: test only checks that some kind of training happened by asserting that the weights are different

        predictor = predictors.FCPredictor(setup_file=base_path / 'test_setup.yaml')

        weights = predictor.model.layers[1].get_weights()[0]

        predictor.train_data = predictor.train_data.iloc[0:200]

        predictor.fit(epochs=2, batch_size=50)

        weights_new = predictor.model.layers[1].get_weights()[0]

        with pytest.raises(AssertionError):
            np.testing.assert_almost_equal(weights, weights_new)


    def test_predict(self, root_dir):

        # FIXME: for now only check that the predict function returns the correct format, don't check actual predictions

        data = pd.read_csv(base_path / 'BesanconGalactic_summary.txt').iloc[0:10]

        assert os.path.isfile(os.path.join(root_dir, 'test_model_FC.h5'))

        model_path = os.path.join(root_dir, 'test_model_FC.h5')
        predictor = predictors.FCPredictor(saved_model=model_path)

        res = predictor.predict(data=data)

        # check that the dimensions are correct
        assert res.shape[0] == 10, "expected 10 predicted rows, got: {}.\n".format(res.shape[0]) +\
                                   " all data:\n{}".format(res)
        assert res.shape[1] == 4, "expected 4 predicted columns, got: {}.\n".format(res.shape[1]) + \
                                  " all data:\n{}".format(res)

        # check the columns:
        for ypar in 'Pfinal    qfinal product   binary_type'.split():
            assert ypar in res.columns, "Expected {} in columns, only".format(ypar) + \
                                         " got: {}.\nall data:\n{}".format(list(res.columns), res)
