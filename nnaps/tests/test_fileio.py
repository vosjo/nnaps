import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import unittest

from nnaps import fileio

from sklearn import preprocessing

from keras.layers import Dense, Input
from keras.models import Model


class TestProcessorConversion(unittest.TestCase):

    def setUP(self):
        self.data = np.random.normal(10, 3, shape=1000)

    def test_standardScaler(self):
        data = np.random.normal(10, 3, size=10)
        data = np.array([data]).T

        std_scale = preprocessing.StandardScaler()
        std_scale.fit(data)

        processors = {'std_scale': std_scale}

        processor_dict = fileio.processors2dict(processors)

        processors_new = fileio.dict2processors(processor_dict)

        std_scale_new = processors_new['std_scale']

        scaled_data = std_scale.transform(data)
        scaled_data_new = std_scale_new.transform(data)

        self.assertTrue(np.all(scaled_data == scaled_data_new),
                        msg="converted scaler does not transform the same as original scaler")

        scaled_data = std_scale.transform(data)
        inv_data = std_scale_new.inverse_transform(scaled_data)

        np.testing.assert_almost_equal(data, inv_data, err_msg="data transformed by original and inverse transformed" +
                                                               " by loaded scaler does not equal original data.")

    def test_robustScaler(self):
        data = np.random.normal(10, 3, size=100)
        data = np.array([data]).T

        rob_scale = preprocessing.RobustScaler()
        rob_scale.fit(data)

        processors = {'rob_scale': rob_scale}

        processor_dict = fileio.processors2dict(processors)

        processors = fileio.dict2processors(processor_dict)

        rob_scale_new = processors['rob_scale']

        scaled_data = rob_scale.transform(data)
        scaled_data_new = rob_scale_new.transform(data)

        self.assertTrue(np.all(scaled_data == scaled_data_new),
                        msg="loaded scaler does not transform the same as original scaler")

        scaled_data = rob_scale.transform(data)
        inv_data = rob_scale_new.inverse_transform(scaled_data)

        np.testing.assert_almost_equal(data, inv_data, err_msg="data transformed by original and inverse transformed" +
                                                               " by loaded scaler does not equal original data.")

    def test_oneHotEncoder(self):
        data_int = np.random.randint(0, 3, size=10)

        data = np.chararray(data_int.shape, itemsize=5)
        data[:] = 'empty'

        data = np.where(data_int == 0, 'type1', data)
        data = np.where(data_int == 1, 'type2', data)
        data = np.where(data_int == 2, 'type3', data)
        data = np.where(data_int == 3, 'type4', data)

        data = np.array([data]).T

        encoder = preprocessing.OneHotEncoder()
        encoder.fit(data)

        processors = {'encoder': encoder}

        processor_dict = fileio.processors2dict(processors)

        processors = fileio.dict2processors(processor_dict)

        encoder_new = processors['encoder']

        scaled_data = encoder.transform(data)
        scaled_data_new = encoder_new.transform(data)

        self.assertTrue(np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]),
                        msg="loaded encoder does not transform the same as original encoder")
        self.assertTrue(np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]),
                        msg="loaded encoder does not transform the same as original encoder")

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data, err_msg="data transformed by original and inverse transformed by" +
                                                        " loaded encoder does not equal original data.")


class TestSafeLoadModel(unittest.TestCase):

    def test_saveLoad_processors(self):

        data = dict(preprocessor='TestEncoder',
                    kwargs={  # OneHot encoder
                        '_n_values': 'auto',
                        '_categories': 'auto',
                        '_categorical_features': 'all',
                        '_legacy_mode': False,
                        'categories_': [np.array(['type1', 'type2', 'type3'], dtype='<U5')],
                        # Standard Scaler
                        'mean_': np.array([10.26388025]),
                        'var_': np.array([8.39983959]),
                        'scale1_': np.array([2.89824768]),
                        # Robust Scaler
                        'center_': np.array([9.99513811]),
                        'scale2_': np.array([3.99362846]),
                    })

        try:
            fileio.save('test.h5', data)
            data_new = fileio.load('test.h5', unpack=False)
        finally:
            if os.path.isfile('test.h5'):
                os.remove('test.h5')

        keys = list(data['kwargs'].keys())
        keys.remove('categories_')
        for key in keys:
            self.assertEqual(data['kwargs'][key], data_new['kwargs'][key])

        self.assertEqual(data_new['kwargs']['categories_'][0].dtype, '|S5',
                         msg="hdf5 saving check when dealing with arrays of strings:\n" +
                             "When saving a numpy array with strings, the returned type should be '|S..'\n" +
                             "got dtype: {}".format(data_new['kwargs']['categories_'][0].dtype))

        np.testing.assert_equal(data['kwargs']['categories_'][0],
                                np.array(data_new['kwargs']['categories_'][0], dtype='<U5'))

    def test_SaveLoad_OneHotEncoder_dtype_char(self):

        data_int = np.random.randint(0, 3, size=10)

        data = np.chararray(data_int.shape, itemsize=5)
        data[:] = 'empty'

        data = np.where(data_int == 0, 'type1', data)
        data = np.where(data_int == 1, 'type2', data)
        data = np.where(data_int == 2, 'type3', data)
        data = np.where(data_int == 3, 'type4', data)

        data = np.array([data]).T

        encoder = preprocessing.OneHotEncoder()
        encoder.fit(data)

        processors = {'encoder': encoder}

        processor_dict = fileio.processors2dict(processors)

        try:
            fileio.save('test_oneHot.h5', processor_dict)
            processor_dict_new = fileio.load('test_oneHot.h5', unpack=False)
        finally:
            if os.path.isfile('test_oneHot.h5'):
                os.remove('test_oneHot.h5')

        processors_new = fileio.dict2processors(processor_dict_new)

        encoder_new = processors_new['encoder']

        scaled_data = encoder.transform(data)
        scaled_data_new = encoder_new.transform(data)

        self.assertTrue(np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]),
                        msg="loaded encoder does not transform the same as original encoder")
        self.assertTrue(np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]),
                        msg="loaded encoder does not transform the same as original encoder")

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data,
                                err_msg="data transformed by original and inverse transformed by loaded encoder" +
                                        " does not equal original data.")

    def test_SaveLoad_OneHotEncoder_dtype_object(self):

        data_int = np.random.randint(0, 3, size=10)

        data = np.chararray(data_int.shape, itemsize=5)
        data[:] = 'empty'

        data = np.where(data_int == 0, 'type1', data)
        data = np.where(data_int == 1, 'type2', data)
        data = np.where(data_int == 2, 'type3', data)
        data = np.where(data_int == 3, 'type4', data)

        data = np.array([data], dtype='object').T

        encoder = preprocessing.OneHotEncoder()
        encoder.fit(data)

        processors = {'encoder': encoder}

        processor_dict = fileio.processors2dict(processors)

        try:
            fileio.save('test_oneHot.h5', processor_dict)
            processor_dict_new = fileio.load('test_oneHot.h5', unpack=False)
        finally:
            if os.path.isfile('test_oneHot.h5'):
                os.remove('test_oneHot.h5')

        processors_new = fileio.dict2processors(processor_dict_new)

        encoder_new = processors_new['encoder']

        scaled_data = encoder.transform(data)
        scaled_data_new = encoder_new.transform(data)

        self.assertTrue(np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]),
                        msg="loaded encoder does not transform the same as original encoder")
        self.assertTrue(np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]),
                        msg="loaded encoder does not transform the same as original encoder")

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data,
                                err_msg="data transformed by original and inverse transformed by loaded encoder" +
                                        " does not equal original data.")

    def test_saveLoad_model(self):

        # make and train a very small model
        inputs = Input(shape=(2,))
        dense1 = Dense(10, activation='relu', name='FC_1')(inputs)
        dense2 = Dense(5, activation='relu', name='FC_2')(dense1)
        output1 = Dense(1, name='output1')(dense2)
        output2 = Dense(1, name='output2')(dense2)
        output3 = Dense(2, activation='softmax', name='output3')(dense2)

        model = Model(inputs, [output1, output2, output3])

        # v1 = np.random.normal(0, 2, 100)
        # v2 = np.random.normal(0.3, 0.5, 100)
        # v3 = np.random.normal(-0.3, 0.5, 100)

        # X = np.array([v1, v2]).T
        # y = [v3, v3]

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        # history = model.fit(X, y, epochs=1, batch_size=20, shuffle=True)

        setup = {'Xpars': ['M1', 'qinit', 'Pinit'],
                 'Yregressors': ['qfinal', 'Pfinal'],
                 'Yclassifiers': ['binary_type', 'product']}

        try:
            fileio.safe_model(model, {}, setup, 'test.h5')
            model_new, processors_new, setup_new = fileio.load_model('test.h5')
        finally:
            os.remove('test.h5')

        self.assertTrue(model.to_json() == model_new.to_json())

        self.assertEqual(setup, setup_new)
