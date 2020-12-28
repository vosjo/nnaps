
Predictors and setup
====================

Features, Targets and Preprocessing
-----------------------------------

The features and targets are defined in the same way regardless of which model you choose. Their setup is defined as
follows when using a yaml setup file. It is the same when providing them as a dictionary:

.. code-block:: yaml

    datafile: 'path_to_csv_file'

    features:
       feature_1:
          processor: processor_1
        ...
    regressors:
       target_regressor_1:
          processor: processor_1
          loss: loss_1
        ...
    classifiers:
       target_classifier_1:
          processor: processor_1
          loss: loss_1
        ...

    random_state: 42
    train_test_split: 0.2

.. option:: datafile (str)

    (Optional) The path to the csv file containing the training and testing data. If not provided, you need to provide
    the training data as a Pandas DataFrame to the Predictor constructor.

.. option:: features (list / dict)

    The features to use in training. For each feature it is possible to provide a preprocessor by using the 'processor'
    keyword. If no processor is provided a standard scaler is used.

.. option:: regressors (list / dict)

    A list of the target regressors that the model should fit. For each regressor you can provide a preprocessor with
    the 'processor' keyword and a loss function with the 'loss' keyword. By default no preprocessing will take place.
    If you don't want to include any regressors in the model, provide an empty list: [].

.. option:: classifiers (list / dict)

    A list of the target classifiers that the model should fit. For each classifier  you can provide a preprocessor with
    the 'processor' keyword and a loss function with the 'loss' keyword. By default a OneHotEncoder is used as
    preprocessor. If you don't want to include any classifiers in the model, provide an empty list: [].

.. option:: random_state (int)

    (Optional) Set the random state of numpy for the train/test split. Can be used for reproducibility.

.. option:: train_test_split (float)

    (Optional) The faction of the provided data to be used as test data. Defaults to 20%.

For each feature you can provide a preprocessor (scaler). If non is explicitly set, the default scaler will be used on
the input parameters. When using the Tree based XGBoost method it is not necessary to scale the input parameters.

For each target a preprocessor and a loss function can be defined. For the regressors there is by default no
preprocessor defined, for the classifiers the default is the one hot encoder. The default regressor loss is 'mean
square error' and the default classifier loss is 'categorical cross entropy'.

The default preprocessors and loss functions can be obtained from defaults:

.. code-block:: python

    from nnaps import defaults

    # default processors for features
    defaults.default_scaler

    # default processor for classifiers
    defaults.default_encoder

    # default regressor loss
    defaults.default_regressor_loss

    # default classifier loss
    defaults.default_classifier_loss


The recognized preprocessors are:

- StandardScaler
- RobustScaler
- MinMaxScaler
- MaxAbsScaler
- OneHotEncoder
- OrdinalEncoder

The recognized loss functions are defined in keras.



Gradient boosted random forest predictor (GBPredictor)
------------------------------------------------------

Setup
^^^^^

Methods
^^^^^^^

.. autoclass:: nnaps.predictors.GBPredictor
    :members:
    :inherited-members:

Fully connected neural network predictor (FCPredictor)
------------------------------------------------------

The FCPredictor consists of a fully connected neural network. You have a lot of freedom on how to define the network
architecture, as long as it is a sequential network (can be interpreted by keras.Sequential. For a full description of
the options, lets have a look at the setup file:

Setup
^^^^^

.. code-block:: yaml

    model:
       - {'layer':'Dense',   'args':[100], 'kwargs': {'activation':'relu', 'name':'FC_1'} }
       - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_1'} }
       - {'layer':'Dense',   'args':[75],  'kwargs': {'activation':'relu', 'name':'FC_2'} }
       - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_2'} }
       - {'layer':'Dense',   'args':[50],  'kwargs': {'activation':'relu', 'name':'FC_3'} }
       - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_3'} }

    optimizer: 'adam'
    optimizer_kwargs: {'learning_rate':0.001, 'beta_1':0.9, 'beta_2':0.999}
    batch_size: 128

.. option:: model (list)

    The architecture of the model. The default architecture consists of 3 fully connected layers with 100, 75 and 50
    nodes, interspaced by dropout regularization layers. The activation function is relu. For now the recognized layers
    are 'Dense' and 'Dropout'.

.. option:: optimizer (str)

    The optimizer to use. Choose from 'sgd', 'adagrad', 'adadelta', 'rmsprop' or 'adam'.

.. option:: optimizer_kwargs (str)

    Any keywords to pass when creating the optimizer.

.. option:: batch_size (int)

    Batch size to use when training the model.

Methods
^^^^^^^

.. autoclass:: nnaps.predictors.FCPredictor
    :members:
    :inherited-members:
