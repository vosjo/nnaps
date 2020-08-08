
Training ML models
==================

When you have processed your grid of MESA models and have obtained a 2D table with your input parameters (variables)
and the targets that you want to predict, it is time to start training a ML model to do population predictions.


Data structure
--------------

NNaPS requires tabular training data, which does not contain NaN values. Internally a pandas DataFrame is used to store
the training data. Training data can be read from a csv file (using the pd.read_csv) function or be provided as a
DataFrame directly to the predictor constructor function. The variables / features are expected to be numerical, while
the targets can be both numerical (continuous or discrete) or categorical variables. And example dataset included in
the package is the 'tests/BesanconGalactic_summary.txt' dataset:

.. code-block:: python

    import pandas as pd

    data = pd.read_csv('tests/BesanconGalactic_summary.txt')

    print(data.head())


======= ============ ========== =========== ============ ========== ========= ==============
   M1     Pinit       qinit      FeHinit     Pfinal       qfinal     product   binary_type
======= ============ ========== =========== ============ ========== ========= ==============
 0.744   134.470005   1.095729   -0.912521   294.031588   0.608444   He-WD     single-lined
 0.813   225.000014   2.524845   -0.806781   153.634007   1.031585   He-WD     single-lined
 0.876   111.550009   2.190000   -0.918768   104.970587   0.912802   He-WD     single-lined
 0.890   512.700045   2.386059   -0.878982   394.729424   1.396449   HB        single-lined
 0.893   102.630007   1.485857   -0.731017   228.613065   0.640067   He-WD     double-lined
======= ============ ========== =========== ============ ========== ========= ==============

The features or input variables here are: M1, Pinit, qinit and FeHinit

The targets that we want to predict are Pfinal, qfinal and product. The first two are continuous numerical targets,
while the last two are categorical variables which we will have to convert.

Model setup
-----------

Now we have our data we can setup the model that we want to use. There are two predictors included in NNaPS: XGBoost
and fully connected neural networks. XGBoost is a very efficient random forest method. The fully connected NNs are
implemented using Keras and TensorFlow. More info about these two types of models can be found in :doc:`nn_predictors`

Here we will use a NN predictor. There are two ways of providing the required setup to the predictor. You can provide
a yaml file detailing the setup, or provide the setup as a dictionary.

.. code-block:: python

    from nnaps import predictors

    # use a setup from file
    predictor = predictors.FCPredictor(setup_file='setup.yaml')

    # use a setup dictionary
    predictor = predictors.FCPredictor(setup=setup_dictionary)

The training data can be provided as a Pandas DataFrame directly to the constructor, or you can provide the filepath
in the setup (both the dictionary or in the setup file). If you want to provide the data directly, use the data keyword.

The most simple setup will consist of the features, the targets and potentially the filepath to the training data:

.. code-block:: yaml

    datafile: 'tests/BesanconGalactic_summary.txt'
    features:
       - M1
       - qinit
       - Pinit
       - FeHinit
    regressors:
       - Pfinal
       - qfinal
    classifiers:
       - product
       - binary_type

You can setup pre processors for both the features and targets, but this is not necessary. If you don't provide them,
NNaPS will add defaults where necessary. For more info on the available options, see :doc:`nn_predictors`.

Model training and predicting
-----------------------------

Training the model is as simple as calling the fit method with potential arguments specific to the model you have
chosen. Predicting new targets can be done with the predict function

.. code-block:: python

    predictor.fit(epochs=100)

    predictions = predictor.predict(new_data)

