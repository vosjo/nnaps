# Neural Network assisted Population Synthesis code

NNAPS aims to be a simple and easy to use codebase for building a population synthesis code from a set of 1D 
stellar/binary evolution models. For example, with NNAPS, a set of MESA models can be turned into a population 
synthesis code. 

NNAPS trains a neural network to act as an interpolator in the provided models, and can then be used to predict 
new models, as long as their starting parameters are in the same range as those used to train the network. 

Lets look at an example:

![wide sdB example](https://raw.githubusercontent.com/vosjo/nnaps/master/docs/wide_sdB_period_q_example.png)

The orbital period and mass ratio distribution of wide hot subdwarf binaries shows a very strong correlation. We want 
to study this correlation and predict the P-q distribution of wide sdB binaries in the galaxy. Using MESA, 2000 models 
covering the observed parameter range were calculated (blue circles in the left figure). Using NNAPS, these models 
were used to train a model and predict the P-q distribution of ~1\,000\,000 binaries (right figure). With this new
BPS model we can now further explore the P-q distribution of different sub populations in our galaxy.

## Simplest use

NNAPS requires a setup file or setup dictionary telling it what to do. The minimaly necessary setup includes a list 
of features and targets together with the path to the training data. Using the test data sample, the simplest model 
setup file in yaml format is the following:

```yaml
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
```

**datafile**: path to the file containing the training data.  This file is read with the pandas.read_csv() function, 
and should be structured in an appropriate way.  
**features**: a list of the features to use when predicting a model. Think of these as the X parameter of your 
prediction function.  
**regressors**: a list of the numerical targets for the model. These are the Y parameters of the prediction function. 
A regressor has to he a continuous numerical value. For a categorical numerical value use a classifier.  
**classifiers**: a list of the categorical targets for the model. These are the Y parameters that are not numerical, 
or the not continuous. 

You can now make and fit the model:

```python
from nnaps import predictors
    
predictor = predictors.BPS_predictor(setup_file='test_setup.yaml')
    
predictor.fit(epochs=100)
```    

create the predictor using a setupfile, and then train it on the provided data. The number of epochs is the number of
iterations to be used in the gradient descent learning. After learning you can check the report if the model is good
enough, or if more iterations are necessary.

Predicting new models is then as simple as providing a pandas DataFrame with the features (X parameters) of the new
models you want to predict to the BPS_predictor:

```python  
new_predictions = predictor.predict(data=new_data)
```    

new_predictions is a pandas dataframe with the predictions and features of the new_data dataframe.

The trained model can be saved to hdf5 format and loaded again for later use:

```python  
predictor.save_model('model.h5')

predictor.load_model('model.h5')
``` 


## Advanced use

It is possible to define many more setting in the setup file. A complete setupfile would look like:

```yaml
datafile: 'tests/BesanconGalactic_summary.txt'

features:
   M1:
      processor: StandardScaler
   qinit:
      processor: StandardScaler
   Pinit:
      processor: StandardScaler
   FeHinit:
      processor: StandardScaler
regressors:
   Pfinal:
      processor: RobustScaler
   qfinal:
      processor: MinMaxScaler
classifiers:
   product: 
      processor: OneHotEcoder
   binary_type:
      processor: OneHotEcoder

random_state: 42
train_test_split: 0.2

model:
   - {'layer':'Dense',   'args':[100], 'kwargs': {'activation':'relu', 'name':'FC_1'} }
   - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_1'} }
   - {'layer':'Dense',   'args':[75],  'kwargs': {'activation':'relu', 'name':'FC_2'} }
   - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_2'} }
   - {'layer':'Dense',   'args':[50],  'kwargs': {'activation':'relu', 'name':'FC_3'} }
   - {'layer':'Dropout', 'args':[0.1], 'kwargs': {'name':'DO_3'} }

optimizer: 'adam'
batch_size: 128
```

## Reports
TODO 



