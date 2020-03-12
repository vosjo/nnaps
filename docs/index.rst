.. nnaps documentation master file, created by
sphinx-quickstart on Tue Feb  4 11:35:28 2020.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

NNaPS
=====

Why a new ML package
--------------------

Machine learning models are already easy to set up with scikit learn or Keras. Why a new package that builds
on top of those? Many astronomers are not acustomed to work with machine learning code. This package aims to 
make it easy to make a model from scratch by providing only the most basic information in a text file and 
letting the code deal with the rest. 
Furthermore, NNaPS allows you to store all necessary information to make, train and use a model in one file. 
Simplifying distributing and checking a model. 

Basic Usage
-----------

Say you have a csv file containing the starting parameter of a set of MESA models together with the observables
that you are interested in. You can make a model predicting those observables in a few lines:

.. code-block:: python

   from nnaps import predictors
   
   setup = {
      'datafile': <path to csv file>,
      'features': ['donor_mass', 'initial_period', 'initial_q'],
      'regressors': ['final_period', 'final_q'],
   }
    
   predictor = predictors.XGBPredictor(setup=setup)
    
   predictor.fit()

   new_predictions = predictor.predict(new_data)


.. toctree::
    :maxdepth: 2
    :caption: User Guide

    userguide/install
    userguide/mesa
    userguide/mesa_2h5
    userguide/mesa_extract
    userguide/predictors
    userguide/reports

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/quickstart
    tutorials/observables
    tutorials/tracks


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

