import yaml

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import utils, metrics
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Dropout
from keras.models import Model

from nnaps import fileio, defaults, plotting


class BPS_predictor():

    def __init__(self, setup=None, setup_file=None, saved_model=None):

        self.processors = None
        self.history = None
        self.setup = None

        self.features = []
        self.regressors = []
        self.classifiers = []

        self.train_data = None
        self.test_data = None

        if not setup is None:
            self.make_from_setup(setup)

        elif not setup_file is None:
            self.make_from_setup_file(setup_file)

        elif not saved_model is None:
            self.load_model(saved_model)


    # { Learning and predicting

    def _append_to_history(self, history):

        # convert the history object to a dataframe
        keys = [c + '_mae' for c in self.regressors]
        keys += ['val_' + c + '_mae' for c in self.regressors]
        keys += [c + '_accuracy' for c in self.classifiers]
        keys += ['val_' + c + '_accuracy' for c in self.classifiers]
        keys += [c + '_loss' for c in self.classifiers + self.regressors]
        keys += ['val_' + c + '_loss' for c in self.classifiers + self.regressors]

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

    def _process_features(self, data):
        """
        Private method. Should NOT be called by user.

        Takes the features from the data frame and runs the required preprocessors on them.
        Assumes that features are always scaled.

        FIXME: allow for features to be not scaled.

        :param data:
        :return:
        """
        X = np.array([self.processors[x].transform(data[[x]]) for x in self.features])
        X = X.reshape(X.shape[:-1]).T
        return X

    def _process_targets(self, data, inverse=False):
        """
        Private method. Should NOT be called by user.

        Takes the targets from the data frame and runs the required preprocessors on them.
        If no preprocessors are set, the original data is returned.


        :param data: target data to be tranformed,
        :param inverse: if true, do the inverse_transform.
        :return: if inverse: a dataframe, else: a numpy array
        """
        if not inverse:
            Y = []
            for y in self.regressors + self.classifiers:
                # check if Y data needs to be transformed before fitting.
                if self.processors[y] is not None:
                    Y.append(self.processors[y].transform(data[[y]]))
                else:
                    Y.append(data[[y]].values)
        else:
            Y = {}
            for Y_, name in zip(data, self.regressors + self.classifiers):
                if self.processors[name] is not None:
                    Y[name] = self.processors[name].inverse_transform(Y_)[:, 0]
                else:
                    Y[name] = Y_[:, 0]
            Y = pd.DataFrame(Y)

        return Y

    def fit(self, data=None, epochs=100, batch_size=128):
        """
        Train the model

        :param data: data to be used for training (pandas DataFrame). Should contain features and targets
        :param epochs: Number of epochs (complete training set coverage) to run.
        :param batch_size: Number of mini-batches to subdivide each epoch in.
        :return: Nothing
        """

        if data is None:
            data = self.train_data

        X = self._process_features(data)
        X_val = self._process_features(self.test_data)

        Y = self._process_targets(data)
        Y_val = self._process_targets(self.test_data)

        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 validation_data=(X_val, Y_val), verbose=2)

        self._append_to_history(history.history)

        train_score = self.score(data)
        test_score = self.score(self.test_data)

        # print the scores
        print("Training results\n{:12s}  {}   {}".format('target', 'training score', 'test score'))
        print("-----------------------------------------")
        for par in self.regressors:
            print( "{:12s}:  {:7.3f}      {:7.3f}".format(par, train_score[par], test_score[par]) )
        for par in self.classifiers:
            print( "{:12s}:  {:6.1f}%      {:6.1f}%".format(par, train_score[par]*100., test_score[par]*100.) )

    def predict(self, data=None):
        """
        Make predictions based on a trained model.

        :param data: the features that you want to use in the prediction. (pandas DataFrame)
        :return: predicted targets for features
        """

        if data is None:
            pass

        X = self._process_features(data)

        Y = self.model.predict(X)

        res = self._process_targets(Y, inverse=True)

        return res

    def score(self, data=None, regressor_metric='mean_absolute_error', classifier_metric='accuracy'):

        if data is None:
            data = self.train_data

        res = self.predict(data)

        scores = {}
        for par in self.regressors:
            score = metrics.mean_absolute_error(data[par], res[par])
            scores[par] = score

        for par in self.classifiers:
            score = metrics.accuracy_score(data[par], res[par])
            scores[par] = score

        return scores

    # }

    # ----------------------------------------------------------------------

    # { Reporting

    def make_training_history_report(self, filename):

        plotting.plot_training_history_html(self.history, targets=self.regressors + self.classifiers,
                                            filename=filename)

    def make_training_data_report(self, filename):

        plotting.plot_training_data_html(self.train_data, self.test_data, self.features,
                                         self.regressors, self.classifiers, self.processors, filename=filename)

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
        Private method. Should NOT be called by user.

        Make the preprocessors from the setup file
        this is required to run before the make_model_from_setup step.

        processors are fitted on the training data only.
        """

        processors = {}

        for pname in self.features:
            p = self.setup['features'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.regressors:
            p = self.setup['regressors'][pname]['processor']
            if p is not None:
                p = p()
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.classifiers:
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

        inputs = Input(shape=(len(self.features),))
        prev_layer = inputs

        # run over all requested layers and connect them
        for layer in model_setup:
            new_layer = get_layer(layer)(prev_layer)
            prev_layer = new_layer

        outputs = []

        for name in self.regressors:
            out = Dense(1, name=name)(prev_layer)
            outputs.append(out)

        for name in self.classifiers:
            num_unique = len(self.processors[name].categories_[0])
            out = Dense(num_unique, activation='softmax', name=name)(prev_layer)
            outputs.append(out)

        self.model = Model(inputs, outputs)

        loss = ['mean_squared_error' for name in self.regressors] + \
               ['categorical_crossentropy' for name in self.classifiers]

        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'mae'])

    def make_from_setup(self, setup):

        self.setup = defaults.add_defaults_to_setup(setup)

        self.features = list(self.setup['features'].keys())
        self.regressors = list(self.setup['regressors'].keys())
        self.classifiers = list(self.setup['classifiers'].keys())

        self._prepare_data()
        self._make_preprocessors_from_setup()
        self._make_model_from_setup()

    def make_from_setup_file(self, filename):

        setupfile = open(filename)
        setup = yaml.safe_load(setupfile)
        setupfile.close()

        self.make_from_setup(setup)

    def save_model(self, filename, include_history=False):
        """
        Save a trained model to hdf5 file for later use
        """
        history = self.history if include_history else None

        fileio.safe_model(self.model, self.processors, self.features, self.regressors, self.classifiers,
                          self.setup, filename, history=history)

    def load_model(self, filename):
        """
        Load a model saved to hdf5 format
        """

        model, processors, features, regressors, classifiers, setup, history = fileio.load_model(filename)
        self.model = model
        self.processors = processors
        self.history = history

        # TODO: not sure if the add_defaults_to_setup should be run automatically
        self.setup = setup

        self.features = features
        self.regressors = regressors
        self.classifiers = classifiers

        # load the data and split it in a training - test set.
        self._prepare_data()

    def save_training_history(self, filename):
        """
        Save the training history to csv file
        """
        self.history.to_csv(filename)

    # }
