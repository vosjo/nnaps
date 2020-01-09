import os

import pandas as pd
import numpy as np

import unittest

from sklearn import preprocessing

from nnaps import predictors

from pathlib import Path

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_path = Path(__file__).parent


class TestBPSPredictorSetup(unittest.TestCase):

    def test_make_from_setup(self):

        predictor = predictors.BPS_predictor(setup_file=base_path / 'test_setup.yaml')
        Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
        Yregressors_ = ['Pfinal', 'qfinal']
        Yclassifiers_ = ['product', 'binary_type']

        # test reading the X and Y variables
        self.assertTrue(all([a in predictor.features for a in Xpars_]))
        self.assertTrue(all([a in predictor.regressors for a in Yregressors_]))
        self.assertTrue(all([a in predictor.classifiers for a in Yclassifiers_]))

        # test making the preprocessors
        pp = predictor.processors

        for par in Xpars_:
            self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
            self.assertTrue(pp[par].__class__ == preprocessing.StandardScaler,
                            msg="{} does not have the correct preprocessor.".format(par) +
                                " expected {}, got {}".format(preprocessing.StandardScaler, pp[par].__class__))

        for par in Yregressors_:
            self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
            self.assertEqual(pp[par], None, msg="{} does not have the correct preprocessor.".format(par) +
                                                " expected None, got {}".format(pp[par].__class__))

        for par in Yclassifiers_:
            self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
            self.assertTrue(pp[par].__class__ == preprocessing.OneHotEncoder,
                            msg="{} does not have the correct preprocessor.".format(par) +
                                " expected {}, got {}".format(preprocessing.OneHotEncoder,pp[par].__class__))

        # test making the model
        mod = predictor.model

        self.assertEqual(len(mod.layers), 11,
                         msg="Model does not have correct number of layers, expected" +
                             " {}, got {}".format(11, len(mod.layers)))

        self.assertEqual(mod.input.shape[1], len(Xpars_),
                         msg="Model input does not have correct shape, expected " +
                            "{}, got {}".format(len(Xpars_),mod.input.shape[1]))

        self.assertEqual(len(mod.output), len(Yregressors_) + len(Yclassifiers_),
                         msg="Model does not have correct number of outputs, expected" +
                             " {}, got {}".format(len(Yregressors_) + len(Yclassifiers_), len(mod.output)))

    def test_make_from_saved_model(self):

        predictor = predictors.BPS_predictor(saved_model=base_path / 'test_model.h5')

        Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
        Yregressors_ = ['Pfinal', 'qfinal']
        Yclassifiers_ = ['product', 'binary_type']

        # test reading the X and Y variables
        self.assertTrue(all([a in predictor.features for a in Xpars_]))
        self.assertTrue(all([a in predictor.regressors for a in Yregressors_]))
        self.assertTrue(all([a in predictor.classifiers for a in Yclassifiers_]))

        # test making the preprocessors
        pp = predictor.processors

        for par in Xpars_:
            self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
            self.assertTrue(pp[par].__class__ == preprocessing.StandardScaler,
                            msg="{} does not have the correct preprocessor. ".format(par) +
                                "expected {}, got {}".format(preprocessing.StandardScaler, pp[par].__class__))

        # for par in Yregressors_:
        #     self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
        #     self.assertTrue(pp[par].__class__ == preprocessing.RobustScaler,
        #                     msg="{} does not have the correct preprocessor. ".format(par) +
        #                         "expected {}, got {}".format(preprocessing.RobustScaler, pp[par].__class__))

        for par in Yclassifiers_:
            self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
            self.assertTrue(pp[par].__class__ == preprocessing.OneHotEncoder,
                            msg="{} does not have the correct preprocessor. ".format(par) +
                                "expected {}, got {}".format(preprocessing.OneHotEncoder, pp[par].__class__))

        # test making the model
        mod = predictor.model

        self.assertEqual(len(mod.layers), 11,
                         msg="Model does not have correct number of layers, expected {}, got {}".format(11, len(
                             mod.layers)))

        self.assertEqual(mod.input.shape[1], len(Xpars_),
                         msg="Model input does not have correct shape, expected {}, got {}".format(len(Xpars_),
                                                                                                   mod.input.shape[1]))

        self.assertEqual(len(mod.output), len(Yregressors_) + len(Yclassifiers_),
                         msg="Model does not have correct number of outputs, expected {}, got {}".format(
                             len(Yregressors_) + len(Yclassifiers_), len(mod.output)))


class TestBPSPredictorTrainingPredicting(unittest.TestCase):

    def test_process_features(self):

        data = np.random.normal(10, 3, size=100)
        df = pd.DataFrame(data={'M1':data, 'qinit':data})

        p = preprocessing.StandardScaler()
        p.fit(df[['M1']])

        processors = dict(M1=p,
                         qinit = p,)

        predictor = predictors.BPS_predictor()

        predictor.processors = processors
        predictor.features = ['M1', 'qinit']

        X = predictor._process_features(df)

        # check that the output shape is correct
        self.assertEqual(X.shape, (100,2))

        # check that all arrays are transformed
        self.assertAlmostEqual(np.mean(X[:,0]), 0)
        self.assertAlmostEqual(np.std(X[:,0]), 1)
        self.assertAlmostEqual(np.mean(X[:,1]), 0)
        self.assertAlmostEqual(np.std(X[:,0]), 1)

    def test_process_targets(self):
        data = np.random.normal(10, 3, size=100)
        df = pd.DataFrame(data={'M1final': data, 'qfinal': data})

        p = preprocessing.StandardScaler()
        p.fit(df[['M1final']])

        processors = dict(M1final=p,
                          qfinal = None,)

        predictor = predictors.BPS_predictor()

        predictor.processors = processors
        predictor.regressors = ['M1final', 'qfinal']

        Y = predictor._process_targets(df)

        # check that the output is a list
        self.assertEqual(type(Y), list)

        # check that both parameters are included
        self.assertEqual(len(Y), 2)

        # check that one array is transformed and the other not
        self.assertAlmostEqual(np.mean(Y[0]), 0)
        self.assertAlmostEqual(np.std(Y[0]), 1)
        np.testing.assert_array_equal(Y[1], df[['qfinal']].values)

    def test_append_to_history(self):

        predictor = predictors.BPS_predictor()

        predictor.regressors = ['M1final']
        predictor.classifiers = []

        data = {'M1final_mae': [0.3, 0.2], 'val_M1final_mae': [0.31, 0.21], 'M1final_loss': [1.5, 1.3],
                'val_M1final_loss': [1.6, 1.4], 'training_run': [1, 1]}
        history1 = pd.DataFrame(data=data)
        history1.index.name = 'epoch'

        predictor.history = history1

        data = {'M1final_mae': [0.1, 0.0], 'val_M1final_mae': [0.11, 0.01], 'M1final_loss': [1.2, 1.1],
                'val_M1final_loss': [1.3, 1.2]}
        history2 = pd.DataFrame(data=data)
        history2.index.name = 'epoch'

        data = {'M1final_mae': [0.3, 0.2, 0.1, 0.0], 'val_M1final_mae': [0.31, 0.21, 0.11, 0.01],
                'M1final_loss': [1.5, 1.3, 1.2, 1.1], 'val_M1final_loss': [1.6, 1.4, 1.3, 1.2],
                'training_run': [1, 1, 2, 2]}
        history_expected = pd.DataFrame(data=data)
        history_expected.index.name = 'epoch'

        predictor._append_to_history(history2)

        history = predictor.history

        self.assertTrue(history.equals(history_expected),
                        msg="\nExpected dataframe: \n{}\nGot dataframe: \n {}".format(history_expected.to_string(),
                                                                                      history.to_string()))

    def test_train_model(self):

        # WARNING: test only checks that some kind of training happened by asserting that the weights are different

        predictor = predictors.BPS_predictor(setup_file=base_path / 'test_setup.yaml')

        weights = predictor.model.layers[1].get_weights()[0]

        predictor.train_data = predictor.train_data.iloc[0:200]

        predictor.fit(epochs=2, batch_size=50)

        weights_new = predictor.model.layers[1].get_weights()[0]

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(weights, weights_new)


    def test_predict(self):

        # FIXME: for now only check that the predict function returns the correct format, don't check actual predictions

        data = pd.read_csv(base_path / 'BesanconGalactic_summary.txt').iloc[0:10]

        predictor = predictors.BPS_predictor(saved_model=base_path / 'test_model.h5')

        res = predictor.predict(data=data)

        # check that the dimensions are correct
        self.assertEqual(res.shape[0], 10, msg="expected 10 predicted rows, got: {}.\n".format(res.shape[0]) +
                                               " all data:\n{}".format(res))
        self.assertEqual(res.shape[1], 4, msg="expected 4 predicted columns, got: {}.\n".format(res.shape[1]) +
                                              " all data:\n{}".format(res))

        # check the columns:
        for ypar in 'Pfinal    qfinal product   binary_type'.split():
            self.assertTrue(ypar in res.columns, msg="Expected {} in columns, only".format(ypar) +
                                                     " got: {}.\nall data:\n{}".format(list(res.columns), res))
