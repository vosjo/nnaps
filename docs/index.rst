NNaPS
=====

What is NNaPS?
--------------

Neural Network assisted Population Synthesis is a python package that wants to make it easy to perform population
synthesis studies with 1D stellar evolution codes. NNaPS is develope specifically with the
`MESA <http://mesa.sourceforge.net/>`_ (Modules and Experiments in Stellar Evolution) in mind. NNaPS provides a MESA
module that simplifies extracting interesting parameters from a grid of MESA runs and a machine learning module that
can be used to create a predictive model based on the results of a MESA run. This model can then be used to perform
a population synthesis study.

A typical experiment will have the following steps:

1. Create a grid of MESA models covering the input parameter space you want to study
2. Use nnaps-mesa to extract the parameters of interest from the MESA grid, and apply CE ejection if wanted
3. Use the machine learning part nnaps.predictors to create a predictive model linking the input parameters of your
   grid to the output parameters of interest
4. Create an input population of a few million models and run them through your predictive model to perform the
   population synthesis study.

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
    userguide/mesa_common_envelope
    userguide/mesa_evolution_phases
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

