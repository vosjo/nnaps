import yaml

import pandas as pd
import numpy as np

from sklearn import preprocessing
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

    def train(self, data=None, epochs=100, batch_size=128, validation_split=0.2):
        """
        Train the model

        :param data:
        :param epochs:
        :param batch_size:
        :param validation_split:
        :return: Nothing
        """

        if data is None:
            data = self.data

        X = np.array([self.processors[x].transform(data[[x]]) for x in self.Xpars])
        X = X.reshape(X.shape[:-1]).T

        Y = []
        for x in self.Yregressors + self.Yclassifiers:
            # check if Y data needs to be transformed before fitting.
            if self.processors[x] is not None:
                Y.append(self.processors[x].transform(data[[x]]))
            else:
                Y.append(data[[x]])

        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 validation_split=validation_split)

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

    def _make_preprocessors_from_setup(self):
        """
        Make the preprocessors from the setup file
        this is required to run before the make_model_from_setup step.
        """

        processors = {}

        print (self.setup)

        for pname in self.Xpars:
            p = self.setup['features'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.data[[pname]])
            processors[pname] = p

        for pname in self.Yregressors:
            p = self.setup['regressors'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.data[[pname]])
            processors[pname] = p

        for pname in self.Yclassifiers:
            p = self.setup['classifiers'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.data[[pname]])
            processors[pname] = p

        self.processors = processors

    def _make_model_from_setup(self):
        """
        Make a model based on a setupfile
        """

        inputs = Input(shape=(len(self.Xpars),))
        dense1 = Dense(100, activation='relu', name='FC_1')(inputs)
        do1 = Dropout(0.1, name='DO_1')(dense1)
        dense2 = Dense(50, activation='relu', name='FC_2')(do1)
        do2 = Dropout(0.1, name='DO_2')(dense2)
        dense3 = Dense(25, activation='relu', name='FC_3')(do2)
        do3 = Dropout(0.1, name='DO_3')(dense3)

        outputs = []

        for name in self.Yregressors:
            out = Dense(1, name=name)(do3)
            outputs.append(out)

        for name in self.Yclassifiers:
            num_unique = len(self.processors[name].categories_[0])
            out = Dense(num_unique, activation='softmax', name=name)(do3)
            outputs.append(out)

        self.model = Model(inputs, outputs)

        loss = ['mean_squared_error' for name in self.Yregressors] + \
               ['categorical_crossentropy' for name in self.Yclassifiers]

        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy', 'mae'])

    def make_from_setup(self, setup):

        self.setup = defaults.add_defaults_to_setup(setup)

        self.Xpars = list(self.setup['features'].keys())
        self.Yregressors = list(self.setup['regressors'].keys())
        self.Yclassifiers = list(self.setup['classifiers'].keys())

        self.data = pd.read_csv(self.setup['datafile'])

        self.optimizer = self.setup.get('optimizer', 'adam')

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
