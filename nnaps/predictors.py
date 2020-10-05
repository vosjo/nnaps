import yaml

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import utils, metrics
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from nnaps import fileio, defaults
from nnaps.reporting import html_reports, pdf_reports

from xgboost import XGBClassifier, XGBRegressor

class BasePredictor():

    def __init__(self):

        self.processors = None
        self.setup = None

        self.features = []
        self.regressors = []
        self.classifiers = []

        self.train_data = None
        self.test_data = None

        self.model = None

        pass

    # { Learning and predicting

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

    def _process_targets(self, data, inverse=False, return_df=False):
        """
        Private method. Should NOT be called by user.

        Takes the targets from the data frame and runs the required preprocessors on them.
        If no preprocessors are set, the original data is returned.


        :param data: target data to be tranformed,
        :param inverse: if true, do the inverse_transform.
        :param return_df: if true, return a dataframe otherwise a numpy array
        :return: the (inverse)transformed targets as array or dataframe
        """

        if isinstance(data, pd.DataFrame):
            # convert dataframe to list. This is necessary because a dataframe can not deal with the
            # output of a softmax classifier
            Y = []
            for name in self.regressors + self.classifiers:
                Y.append(data[name].values.reshape(-1, 1))
            data = Y
        else:
            # if there is only one parameter, make a list of it
            if len(self.regressors+self.classifiers) == 1:
                data = [data]

        # the processors need a 2D array even though they only deal with one feature
        # the dataframe constructor needs 1D arrays

        if return_df:
            Y = {}
            for Y_, name in zip(data, self.regressors + self.classifiers):
                if self.processors[name] is not None:
                    if inverse:
                        Y[name] = self.processors[name].inverse_transform(Y_)[:, 0]
                    else:
                        Y[name] = self.processors[name].transform(Y_)[:, 0]
                else:
                    Y[name] = Y_[:, 0]
            Y = pd.DataFrame(Y)
        else:
            Y = []
            for Y_, name in zip(data, self.regressors + self.classifiers):
                # check if Y data needs to be transformed before fitting.
                if self.processors[name] is not None:
                    if inverse:
                        Y.append(self.processors[name].inverse_transform(Y_))
                    else:
                        Y.append(self.processors[name].transform(Y_))
                else:
                    Y.append(Y_)

        return Y

    def fit(self, data=None):
        raise NotImplementedError("The fit routine must be implemented "
                                  "by subclasses")

    def predict(self, data=None):
        raise NotImplementedError("The predict routine must be implemented "
                                  "by subclasses")

    def score(self, data=None, regressor_metric='mean_absolute_error', classifier_metric='accuracy'):
        """
        Calculates the score of a model on the provided data.

        :param data: The data on which to score the network (DataFrame)
        :param regressor_metric: (optional) which metric to use for regressors
        :param classifier_metric: (optional) which metric to use for classifiers.
        :return: The scores
        """

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

    #}

    # ----------------------------------------------------------------------

    # { Reporting

    def make_training_history_report(self, filename):

        html_reports.make_training_history_report(self, filename=filename)

    def make_training_data_report(self, filename):

        html_reports.make_training_data_report(self, filename=filename)

    def print_score(self, training_data=None, test_data=None):
        """
        prints the scores of the current model on the training and test data.

        Uses the score function to calculate the scores.

        :param training_data: training data set to score, if None the stored one is used.
        :param test_data: set data set to score, is None the stored one is used.
        :return: None
        """

        if training_data is None:
            training_data = self.train_data
        if test_data is None:
            test_data = self.test_data

        train_score = self.score(training_data)
        test_score = self.score(test_data)

        # print the scores
        print("Training results\n{:12s}     {}  {}   {}".format('target', 'mean', 'training score', 'test score'))
        print("--------------------------------------------------")
        for par in self.regressors:
            mean = training_data[par].mean()
            print("{:12s}:  {:7.3f}      {:7.3f}      {:7.3f}".format(par, mean, train_score[par], test_score[par]))
        for par in self.classifiers:
            print("{:12s}:  {}      {:6.1f}%      {:6.1f}%".format(par, '-', train_score[par] * 100.,
                                                                   test_score[par] * 100.))

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix for every classifier included in the model

        :return: nothing
        """

        pdf_reports.plot_confusion_matrix(self)

    def plot_feature_range_comparison(self, data):

        pdf_reports.plot_feature_range_comparison(self, data)

    # }

    # ----------------------------------------------------------------------

    # { Input and output

    def _prepare_data(self, data=None):
        """
        Private method. Should NOT be called by user.

        Reads the data from the path given in the setup ('datafile'),
        randomly shuffles the data, and then divides it into a train and test set using the 'train_test_split'
        fraction given in the setup. If a 'random_state' is defined in the setup, this will be used in both
        the shuffle and the train-test split.

        :param data: a pandas dataframe can be directly provided to the method. In that case it will not read the
                     datafile keyword of the setup
        :return: nothing
        """

        if data is None and 'datafile' in self.setup:
            data = pd.read_csv(self.setup['datafile'])
        elif data is None and 'datafile' not in self.setup:
            return
        data = utils.shuffle(data, random_state=self.setup['random_state'])
        data_train, data_test = train_test_split(data, test_size=self.setup['train_test_split'],
                                                 random_state=self.setup['random_state'])

        self.train_data = data_train
        self.test_data = data_test

    def _make_preprocessors_from_setup(self):
        """
        Private method. Should NOT be called by user.

        Make the preprocessors from the setup file
        this is required to run before the make_model_from_setup step.

        processors are fitted on the training data only.
        """

        # if there is no training data, the processors can't be fitted.
        if self.train_data is None:
            return

        processors = {}

        for pname in self.features:
            p = self.setup['features'][pname]['processor']
            if p is not None:
                p = defaults.get_processor_class(p)
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.regressors:
            p = self.setup['regressors'][pname]['processor']
            if p is not None:
                p = defaults.get_processor_class(p)
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        for pname in self.classifiers:
            p = self.setup['classifiers'][pname]['processor']
            if p is not None:
                p = defaults.get_processor_class(p)
                p.fit(self.train_data[[pname]])
            processors[pname] = p

        self.processors = processors

    #}

class XGBPredictor(BasePredictor):

    def __init__(self, setup=None, setup_file=None, saved_model=None, data=None):
        super().__init__()

        if not setup is None:
            self.make_from_setup(setup, data=data)

        elif not setup_file is None:
            self.make_from_setup_file(setup_file, data=data)

        elif not saved_model is None:
            self.load_model(saved_model)

    # { Learning and predicting

    def fit(self, data=None):
        """
        Train the model

        :param data: data to be used for training (pandas DataFrame). Should contain features and targets
        :return: Nothing
        """

        if data is None:
            data = self.train_data

        X = self._process_features(data)
        X_val = self._process_features(self.test_data)

        Y = self._process_targets(data, return_df=True)
        Y_val = self._process_targets(self.test_data, return_df=True)

        for name in self.regressors + self.classifiers:
            self.model[name].fit(X, Y[name].values)
            print ("fitted model for {}".format(name))


        self.print_score(training_data=data)

    def predict(self, data=None):
        """
        Make predictions based on a trained model.

        :param data: the features that you want to use in the prediction. (pandas DataFrame)
        :return: predicted targets for features
        """

        if data is None:
            pass

        X = self._process_features(data)

        Y = {}
        for name in self.regressors+self.classifiers:
            Y[name] = self.model[name].predict(X)
        Y = pd.DataFrame(Y)

        res = self._process_targets(Y, inverse=True, return_df=True)

        return res


    # }

    # { Input and output

    def _make_model_from_setup(self):
        """
        Prepare the regressor and classifier models and store them in the model variable.
        """

        models = {}
        for name in self.regressors:
            models[name] = XGBRegressor()

        for name in self.classifiers:
            models[name] = XGBClassifier()

        self.model = models

    def make_from_setup(self, setup, data=None):

        self.setup = defaults.add_defaults_to_setup(setup)

        self.features = list(self.setup['features'].keys())
        self.regressors = list(self.setup['regressors'].keys())
        self.classifiers = list(self.setup['classifiers'].keys())

        self._prepare_data(data=data)
        self._make_preprocessors_from_setup()
        self._make_model_from_setup()

    def make_from_setup_file(self, filename, data=None):

        setupfile = open(filename)
        setup = yaml.safe_load(setupfile)
        setupfile.close()

        self.make_from_setup(setup, data=data)

    def save_model(self, filename):
        """
        Save a trained model to hdf5 file for later use
        """

        fileio.safe_model(self.model, self.processors, self.features, self.regressors, self.classifiers,
                          self.setup, filename, history=None, method='pickle')

    def load_model(self, filename):
        """
        Load a model saved to hdf5 format
        """

        model, processors, features, regressors, classifiers, setup, history = fileio.load_model(filename)
        self.model = model
        self.processors = processors

        self.setup = setup

        self.features = features
        self.regressors = regressors
        self.classifiers = classifiers

        # load the data and split it in a training - test set.
        self._prepare_data()

    #}

class FCPredictor(BasePredictor):

    def __init__(self, setup=None, setup_file=None, saved_model=None, data=None):

        super().__init__()

        # to store the fitting history of the FC network
        self.history = None

        if not setup is None:
            self.make_from_setup(setup, data=data)

        elif not setup_file is None:
            self.make_from_setup_file(setup_file, data=data)

        elif not saved_model is None:
            self.load_model(saved_model)


    # { Learning and predicting

    def _append_to_history(self, history):

        if len(self.regressors + self.classifiers) == 1:
            # when only one target KERAS does not add the target name to the history file
            parname = (self.regressors + self.classifiers)[0]

            if len(self.regressors) > 0:
                hist_keys = ['mae', 'val_mae']
                new_keys = [parname+'_mae', 'val_' +parname+'_mae']
            else:
                hist_keys = ['accuracy', 'val_accuracy']
                new_keys = [parname+'_accuracy', 'val_' +parname+'_accuracy']

            hist_keys += ['loss', 'val_loss']
            new_keys += [parname+'_loss', 'val_'+parname+'_loss']
            data = {k1: history[k2] for k1, k2 in zip(new_keys, hist_keys)}

        else:
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
            self.history = self.history.append(history_df, sort=False)

    def fit(self, data=None, epochs=100, batch_size=128, early_stopping=None, reduce_lr=None, min_lr=None, verbose=2):
        """
        Train the model

        :param data: data to be used for training (pandas DataFrame). Should contain features and targets
        :param epochs: Number of epochs (complete training set coverage) to run.
        :param batch_size: Number of mini-batches to subdivide each epoch in.
        :param early_stopping: stop training when validation set has reached the minimum loss
        :param reduce_lr: reduce the learning rate when a plateau is reached in the loss curves
        :param min_lr: the minimum learning rate. if None, the learning rate is allowed to decrease by 4 orders
                       of magnitude.
        :return: Nothing
        """

        if data is None:
            data = self.train_data

        if early_stopping is None:
            early_stopping = self.setup['early_stopping']
        if reduce_lr is None:
            reduce_lr = self.setup['reduce_lr']

        X = self._process_features(data)
        X_val = self._process_features(self.test_data)

        Y = self._process_targets(data)
        Y_val = self._process_targets(self.test_data, return_df=False)

        callbacks = []
        if early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            callbacks = [es]

        if reduce_lr:
            if min_lr is None:
                min_lr = float(self.model.optimizer.lr.value()) / 1e4
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=5, min_lr=min_lr,
                                          verbose=1)
            callbacks.append(reduce_lr)

        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 validation_data=(X_val, Y_val), verbose=verbose, callbacks=callbacks)

        self._append_to_history(history.history)

        self.print_score(training_data=data)

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

        res = self._process_targets(Y, inverse=True, return_df=True)

        return res

    # }

    # ----------------------------------------------------------------------

    # { Reporting

    def plot_training_history(self):
        pdf_reports.plot_training_history(self)

    #}

    # ----------------------------------------------------------------------

    # { Input and output

    def _make_model_from_setup(self):
        """
        Make a model based on information given in the setup
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

        loss = [self.setup['regressors'][name]['loss'] for name in self.regressors] + \
               [self.setup['classifiers'][name]['loss'] for name in self.classifiers]

        model_metrics = [['mae'] for name in self.regressors] + \
                        [['accuracy'] for name in self.classifiers]

        opt = defaults.get_optimizer(self.setup['optimizer'], optimizer_kwargs=self.setup['optimizer_kwargs'])
        self.model.compile(optimizer=opt, loss=loss, metrics=model_metrics)

    def make_from_setup(self, setup, data=None):

        self.setup = defaults.add_defaults_to_setup(setup)

        self.features = list(self.setup['features'].keys())
        self.regressors = list(self.setup['regressors'].keys())
        self.classifiers = list(self.setup['classifiers'].keys())

        self._prepare_data(data=data)
        self._make_preprocessors_from_setup()
        self._make_model_from_setup()

    def make_from_setup_file(self, filename, data=None):

        setupfile = open(filename)
        setup = yaml.safe_load(setupfile)
        setupfile.close()

        self.make_from_setup(setup, data=data)

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
