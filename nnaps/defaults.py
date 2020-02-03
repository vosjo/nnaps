from sklearn import preprocessing

from keras import optimizers

import copy

default_scaler = 'StandardScaler'
default_encoder = 'OneHotEncoder'

default_regressor_loss = 'mean_squared_error'
default_classifier_loss = 'categorical_crossentropy'

default_model = [{'layer':'Dense', 'args':[100], 'kwargs': dict(activation='relu', name='FC_1')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_1')},
                 {'layer':'Dense', 'args':[50], 'kwargs': dict(activation='relu', name='FC_2')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_2')},
                 {'layer':'Dense', 'args':[25], 'kwargs': dict(activation='relu', name='FC_3')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_3')},
                 ]

default_regressor_model = []
default_classifier_model = []

def get_processor_class(processor_name):

    valid_processors = ['OneHotEncoder', 'OrdinalEncoder', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler']
    if not processor_name in valid_processors:
        raise ValueError("Processor {} not recognized as valid Processor.".format(processor_name) +
                         "\nAllowed processors: {}".format(valid_processors))

    if processor_name == 'OneHotEncoder':
        return preprocessing.OneHotEncoder()

    if processor_name == 'OrdinalEncoder':
        return preprocessing.OrdinalEncoder()

    elif processor_name == 'StandardScaler':
        return preprocessing.StandardScaler()

    elif processor_name == 'RobustScaler':
        return preprocessing.RobustScaler()

    elif processor_name == 'MinMaxScaler':
        return preprocessing.MinMaxScaler()

    elif processor_name == 'MaxAbsScaler':
        return preprocessing.MaxAbsScaler()

    else:
        return None

def get_optimizer(optimizer, optimizer_kwargs=None):
    """
    Returns the KERAS optimizer based on the name and optional keyword arguments for that optimizer.
    If the name of the optimizer is not recognized, a ValueError is raised.

    recognized optimizers are:
    - sgd: Stochastic gradient descent
    - adagrad:  Adaptive Subgradient Methods (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    - adadelta: ADADELTA (https://arxiv.org/abs/1212.5701)
    - rmsprop: Root mean square prop
    - adam: adaptive moment estimation (http://arxiv.org/pdf/1412.6980v8.pdf)

    :param optimizer: name of the optimizer
    :param optimizer_kwargs: optional keyword arguments for the optimizer.
    :return: the keras optimizer.
    """

    valid_optimizers = ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']
    if not optimizer in valid_optimizers:
        raise ValueError("Optimizer {} not recognized as valid optimizer in keras.".format(optimizer) +
                         "\nAllowed optimizers: {}".format(valid_optimizers))

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if optimizer == 'sgd':
        return optimizers.SGD(**optimizer_kwargs)
    elif optimizer == 'adagrad':
        return optimizers.Adagrad(**optimizer_kwargs)
    elif optimizer == 'adadelta':
        return optimizers.Adadelta(**optimizer_kwargs)
    elif optimizer == 'rmsprop':
        return optimizers.RMSprop(**optimizer_kwargs)
    elif optimizer == 'adam':
        return optimizers.Adam(**optimizer_kwargs)


def update_with_default_dict(main_dict, level1_key, default_dict, in_place=False):
    """
    takes a dictionary of the form:
    {
    level1_key1:
        level2_key1:
            level3_key1: value1
            level3_key2: value2
        level2_key2:
            level3_key1: value1
            level3_key2: value2
    level1_key2:
        ...
    }
    and updates all dictionary under the given level1 key with the defaults given in default_dict if necessary.
    The level3 key-value pairs are updated with the key-value pairs in the default dict if the level3 dictionary
    is located under the level1 key given.

    :param main_dict: The main dictionary in which to make the changes
    :param level1_key: The key of the dictionary under which all updates will take place
    :param default_dict: the dictionary used to update the level3 dictionaries.
    :param in_place: if True, the changes are made in the given dictionary, else a copy is used.
    :return: Dictionary with the updated values if not in_place else nothing
    """

    if not in_place:
        main_dict = copy.deepcopy(main_dict)

    if type(main_dict[level1_key]) is list:
        level2_new = {}
        for f in main_dict[level1_key]:
            level2_new[f] = default_dict

        main_dict[level1_key] = level2_new

    else:
        for level2_key in main_dict[level1_key].keys():
            if main_dict[level1_key][level2_key] is None:
                # if there is no settings dictionary, just add the default
                main_dict[level1_key][level2_key] = default_dict
            else:
                # if there is a dictionary, update the default dictionary with the provided dictionary
                default_dict_ = default_dict.copy()
                default_dict_.update(main_dict[level1_key][level2_key])
                main_dict[level1_key][level2_key] = default_dict_

    if not in_place: return main_dict

def add_defaults_to_setup(setup):

    setup = copy.deepcopy(setup)

    user_scaler = setup.get('default_scaler', default_scaler)
    user_encoder = setup.get('default_encoder', default_encoder)

    # check and update the preprocessors for features and regressors/classifiers
    setup = update_with_default_dict(setup, 'features', {'processor': user_scaler})
    if 'regressors' in setup:
        setup = update_with_default_dict(setup, 'regressors', {'processor': None,
                                                              'loss': default_regressor_loss})
    else:
        setup['regressors'] = {}
    if 'classifiers' in setup:
        setup = update_with_default_dict(setup, 'classifiers', {'processor': user_encoder,
                                                               'loss': default_classifier_loss})
    else:
        setup['classifiers'] = {}

    # check and update the model setup
    if not 'model' in setup:
        setup['model'] = default_model

    if not 'regressor_model' in setup:
        setup['regressor_model'] = default_regressor_model

    if not 'classifier_model' in setup:
        setup['classifier_model'] = default_classifier_model

    if not 'random_state' in setup: setup['random_state'] = 42
    if not 'train_test_split' in setup: setup['train_test_split'] = 0.2

    if not 'optimizer' in setup:
        setup['optimizer'] = 'adam'
    if not 'optimizer_kwargs' in setup:
        setup['optimizer_kwargs'] = None

    if not 'early_stopping' in setup: setup['early_stopping'] = True
    if not 'reduce_lr' in setup: setup['reduce_lr'] = True

    return setup
