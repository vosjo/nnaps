import os
import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np

from nnaps import fileio

from sklearn import preprocessing

from keras.layers import Dense, Input, Activation
from keras.models import Model, Sequential


class TestProcessorConversion:

    def scaler2dict2scaler_test(self, scaler, data):

        scaler.fit(data)

        processors = {'processor': scaler}

        processor_dict = fileio.processors2dict(processors)

        processors_new = fileio.dict2processors(processor_dict)

        new_scaler = processors_new['processor']

        scaled_data = scaler.transform(data)
        new_scaled_data = new_scaler.transform(data)
        inv_scaled_data = new_scaler.inverse_transform(scaled_data)

        np.testing.assert_array_equal(scaled_data, new_scaled_data,
                                      err_msg="loaded scaler does not transform the same as original scaler")

        np.testing.assert_almost_equal(data, inv_scaled_data, err_msg="data transformed by original and inverse" +
                                                    " transformed by loaded scaler does not equal original data.")


    def test_standardScaler(self):
        data = np.random.normal(10, 3, size=10)
        data = np.array([data]).T

        std_scaler = preprocessing.StandardScaler()

        self.scaler2dict2scaler_test(std_scaler, data)


    def test_robustScaler(self):
        data = np.random.normal(10, 3, size=100)
        data = np.array([data]).T

        rob_scaler = preprocessing.RobustScaler()

        self.scaler2dict2scaler_test(rob_scaler, data)



    def test_minMaxScaler(self):
        data = np.random.normal(10, 3, size=100)
        data = np.array([data]).T

        minmax_scaler = preprocessing.MinMaxScaler()

        self.scaler2dict2scaler_test(minmax_scaler, data)


    def test_maxAbsScaler(self):
        data = np.random.normal(10, 3, size=100)
        data = np.array([data]).T

        maxabs_scaler = preprocessing.MaxAbsScaler()

        self.scaler2dict2scaler_test(maxabs_scaler, data)


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

        assert np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]),\
               "loaded encoder does not transform the same as original encoder"
        assert np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]), \
               "loaded encoder does not transform the same as original encoder"

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data, err_msg="data transformed by original and inverse transformed by" +
                                                        " loaded encoder does not equal original data.")

# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='function')
def test_scaler():

    def test_scaler_helper(scaler):

        data = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

        scaler.fit(data)

        processors = {'scaler': scaler}

        processor_dict = fileio.processors2dict(processors)

        try:
            fileio.save('test_scaler.h5', processor_dict)
            processor_dict_new = fileio.load('test_scaler.h5', unpack=False)
        finally:
            if os.path.isfile('test_scaler.h5'):
                os.remove('test_scaler.h5')

        processors_new = fileio.dict2processors(processor_dict_new)

        scaler_new = processors_new['scaler']

        scaled_data = scaler.transform(data)
        scaled_data_new = scaler_new.transform(data)

        assert np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]), \
            "{}: loaded scaler does not transform the same as original scaler".format(scaler.__class__)
        assert np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]), \
            "{}: loaded scaler does not transform the same as original scaler".format(scaler.__class__)

    return test_scaler_helper


class TestSaveLoadScalers:

    @pytest.mark.usefixtures("test_scaler")
    def test_saveload_standardScaler(self, test_scaler):

        scaler = preprocessing.StandardScaler()
        test_scaler(scaler)

    @pytest.mark.usefixtures("test_scaler")
    def test_saveload_robustScaler(self, test_scaler):

        scaler = preprocessing.RobustScaler()
        test_scaler(scaler)

    @pytest.mark.usefixtures("test_scaler")
    def test_saveload_minMaxScaler(self, test_scaler):

        scaler = preprocessing.MinMaxScaler()
        test_scaler(scaler)

    @pytest.mark.usefixtures("test_scaler")
    def test_saveload_maxAbsScaler(self, test_scaler):

        scaler = preprocessing.MaxAbsScaler()
        test_scaler(scaler)

# ----------------------------------------------------------------------------------------------------------------------


class TestSaveLoadEncoders:

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
                        # MinMax Scaler
                        'min_': np.array([-0.09978182]),
                        'data_min_': np.array([1.69929507]),
                        'data_max_': np.array([18.72940234]),
                        'data_range_': np.array([17.03010727]),
                        # MaxAbs Scaler
                        'max_abs_': np.array([18.72940234]),
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
            assert data['kwargs'][key] == data_new['kwargs'][key]

        assert data_new['kwargs']['categories_'][0].dtype == '|S5', \
            "hdf5 saving check when dealing with arrays of strings:\n" + \
            "When saving a numpy array with strings, the returned type should be '|S..'\n" + \
             "got dtype: {}".format(data_new['kwargs']['categories_'][0].dtype)

        np.testing.assert_equal(data['kwargs']['categories_'][0],
                                np.array(data_new['kwargs']['categories_'][0], dtype='<U5'))

    def test_saveload_onehotencoder_dtype_char(self):

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

        assert np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]), \
            "loaded encoder does not transform the same as original encoder"
        assert np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]), \
            "loaded encoder does not transform the same as original encoder"

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data,
                                err_msg="data transformed by original and inverse transformed by loaded encoder" +
                                        " does not equal original data.")

    def test_saveload_onehotencoder_dtype_object(self):

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

        assert np.all(scaled_data.nonzero()[0] == scaled_data_new.nonzero()[0]), \
            "loaded encoder does not transform the same as original encoder"
        assert np.all(scaled_data.nonzero()[1] == scaled_data_new.nonzero()[1]), \
            "loaded encoder does not transform the same as original encoder"

        scaled_data = encoder.transform(data)
        inv_data = encoder_new.inverse_transform(scaled_data)

        np.testing.assert_equal(data, inv_data,
                                err_msg="data transformed by original and inverse transformed by loaded encoder" +
                                        " does not equal original data.")


class TestSaveLoadModel:

    def test_saveload_model(self):

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

        try:
            fileio.safe_model(model, {}, [], [], [], {}, 'test.h5')
            model_new, _, _, _, _, _, _ = fileio.load_model('test.h5')
        finally:
            os.remove('test.h5')

        assert model.to_json() == model_new.to_json()

    def test_saveload_history(self):

        # the save function NEEDS a model to work
        model = Sequential([
            Dense(32, input_shape=(5,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax'),
        ])

        data = {'M1final_mae': [0.3, 0.2], 'val_M1final_mae': [0.31, 0.21], 'M1final_loss': [1.5, 1.3],
                'val_M1final_loss': [1.6, 1.4], 'training_run': [1, 1]}
        history = pd.DataFrame(data=data)
        history.index.name = 'epoch'

        try:
            fileio.safe_model(model, {}, [], [], [], {}, 'test.h5', history=history)
            _, _, _, _, _, _, history_new = fileio.load_model('test.h5')
        finally:
            os.remove('test.h5')

        np.testing.assert_array_equal(history.columns, history_new.columns)

        np.testing.assert_array_equal(history.values, history_new.values)

        assert history_new.index.name == 'epoch'
