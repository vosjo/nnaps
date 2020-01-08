import yaml

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Dropout
from keras.models import Model

from nnaps import fileio, defaults, plotting


class BPS_predictor():

    def __init__(self, setup=None, setup_file=None, saved_model=None):

        self.history = None
        self.setup = None

        self.Xpars = []
        self.Yregressors = []
        self.Yclassifiers = []

        if not setup is None:
            self.make_from_setup(setup)

        elif not setup_file is None:
            self.make_from_setup_file(setup_file)

        elif not saved_model is None:
            self.load_model(saved_model)


    # { Learning and predicting

    def _append_to_history(self, history):

        # convert the history object to a dataframe
        keys = [c + '_mae' for c in self.Yregressors]
        keys += ['val_' + c + '_mae' for c in self.Yregressors]
        keys += [c + '_accuracy' for c in self.Yclassifiers]
        keys += ['val_' + c + '_accuracy' for c in self.Yclassifiers]
        keys += [c + '_loss' for c in self.Yclassifiers + self.Yregressors]
        keys += ['val_' + c + '_loss' for c in self.Yclassifiers + self.Yregressors]

        data = {k: history[k] for k in keys}

        history_df = pd.DataFrame(data=data)
        history_df.index.name = 'epoch'

        # append to existing history file, or set history file
        if self.history is None:
            history_df['training_run'] = 1
            self.history = history_df

        else:
            history_df.index += len(self.history)
            history_df['training_run'] = np.max(self.history['training_run']) + 1
            self.history = self.history.append(history_df)

    def train(self, data=None, epochs=100, batch_size=128):
        """
        Train the model

        :param data:
        :param epochs:
        :param batch_size:
        :param validation_split:
        :return: Nothing
        """

        if data is None:
            data = self.train_data

        def proces_features(data):
            X = np.array([self.processors[x].transform(data[[x]]) for x in self.Xpars])
            X = X.reshape(X.shape[:-1]).T
            return X

        def process_values(data):
            Y = []
            for y in self.Yregressors + self.Yclassifiers:
                # check if Y data needs to be transformed before fitting.
                if self.processors[y] is not None:
                    Y.append(self.processors[y].transform(data[[y]]))
                else:
                    Y.append(data[[y]])
            return Y

        X = proces_features(data)
        X_val = proces_features(self.test_data)

        Y = process_values(data)
        Y_val = process_values(self.test_data)

        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 validation_data=(X_val, Y_val), verbose=2)

        self._append_to_history(history.history)

    def predict(self, data=None):
        """
        Make predictions based on a trained model

        :param data:
        :return: predicted values for data
        """

        if data is None:
            pass

        X = np.array([self.processors[x].transform(data[[x]]) for x in self.Xpars])
        X = X.reshape(X.shape[:-1]).T

        Y = self.model.predict(X)

        res = {}
        for Y_, name in zip(Y, self.Yregressors + self.Yclassifiers):
            if self.processors[name] is not None:
                res[name] = self.processors[name].inverse_transform(Y_)[:, 0]
            else:
                res[name] = Y_

        return pd.DataFrame(data=res)

    # }

    # ----------------------------------------------------------------------

    # { Plotting

    def plot_training_history(self):

        plotting.plot_training_history_html(self.history)

    # }

    # ----------------------------------------------------------------------

    # { Input and output

    def _prepare_data(self):
        """
        Private method. Should NOT be called by user.

        Reads the data from the path given in the setup ('datafile'),
        randomly shuffles the data, and then divides it into a train and test set using the 'train_test_split'
        fraction given in the setup. If a 'random_state' is defined in the setup, this will be used in both
        the shuffle and the train-test split.

        :return: nothing
        """

        data = pd.read_csv(self.setup['datafile'])
        data = utils.shuffle(data, random_state=self.setup['random_state'])
        data_train, data_test = train_test_split(data, test_size=self.setup['train_test_split'],
                                                random_state=self.setup['random_state'])

        self.train_data =  data_train
        self.test_data = data_test

    def _make_preprocessors_from_setup(self):
        """
        Make the preprocessors from the setup file
        this is required to run before the make_model_from_setup step.

        processors are fitted on the training data only.
        """

        processors = {}

        for pname in self.Xpars:
            p = self.setup['features'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.Yregressors:
            p = self.setup['regressors'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.Yclassifiers:
            p = self.setup['classifiers'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        self.processors = processors

    def _make_model_from_setup(self):
        """
        Make a model based on a setupfile
        """
        # TODO: still needs to add support for extra regressor and classifier layers after the main body of the model

        def get_layer(layer):
            if layer['layer'] == 'Dense':
                return Dense(*layer['args'], **layer['kwargs'])
            elif layer['layer'] == 'Dropout':
                return Dropout(*layer['args'], **layer['kwargs'])

        model_setup = self.setup['model']

        inputs = Input(shape=(len(self.Xpars),))
        prev_layer = inputs

        # run over all requested layers and connect them
        for layer in model_setup:
            new_layer = get_layer(layer)(prev_layer)
            prev_layer = new_layer

        outputs = []

        for name in self.Yregressors:
            out = Dense(1, name=name)(prev_layer)
            outputs.append(out)

        for name in self.Yclassifiers:
            num_unique = len(self.processors[name].categories_[0])
            out = Dense(num_unique, activation='softmax', name=name)(prev_layer)
            outputs.append(out)

        self.model = Model(inputs, outputs)

        loss = ['mean_squared_error' for name in self.Yregressors] + \
               ['categorical_crossentropy' for name in self.Yclassifiers]

        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'mae'])

    def make_from_setup(self, setup):

        self.setup = defaults.add_defaults_to_setup(setup)

        self.Xpars = list(self.setup['features'].keys())
        self.Yregressors = list(self.setup['regressors'].keys())
        self.Yclassifiers = list(self.setup['classifiers'].keys())


        self._prepare_data()
        self._make_preprocessors_from_setup()
        self._make_model_from_setup()

    def make_from_setup_file(self, filename):

        setupfile = open(filename)
        setup = yaml.safe_load(setupfile)
        setupfile.close()

        self.make_from_setup(setup)

    def save_model(self, filename):
        """
      Save a trained model to hdf5 file for later use
      """

        setup = {'Xpars': self.Xpars,
                 'Yregressors': self.Yregressors,
                 'Yclassifiers': self.Yclassifiers}

        fileio.safe_model(self.model, self.processors, setup, filename)

    def load_model(self, filename):
        """
      Load a model saved to hdf5 format
      """

        model, processors, setup = fileio.load_model(filename)
        self.model = model
        self.processors = processors

        self.Xpars = setup['Xpars']
        self.Yregressors = setup['Yregressors']
        self.Yclassifiers = setup['Yclassifiers']

    def save_training_history(self, filename):
        """
      Save the traning history to csv file
      """
        self.history.to_csv(filename)

    # }
