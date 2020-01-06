 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn import preprocessing

from keras.layers import Dense, Input, Dropout
from keras.models import Model

import yaml

import pylab as pl

import fileio


def make_model(Xshape, Yregressors, Yclassifiers, processors):
   # creating model
   
   inputs = Input(shape = (Xshape,))
   dense1 = Dense(100, activation = 'relu', name='FC_1')(inputs)
   dense2 = Dense(50, activation = 'relu', name='FC_2')(dense1)
   dense3 = Dense(25, activation = 'relu', name='FC_3')(dense2)

   outputs = []

   for name in Yregressors:
      out = Dense(1, name=name)(dense3)
      outputs.append(out)

   for name in Yclassifiers:
      num_unique = len(processors[name].categories_[0])
      out = Dense(num_unique, activation = 'softmax', name=name)(dense3)
      outputs.append(out)

   model = Model(inputs, outputs)
   
   return model

def make_preprocessors(data, Xpars, Yregressors, Yclassifiers):
   
   processors = {}
   
   for pname in Xpars:
      p = preprocessing.StandardScaler()
      p.fit(data[[pname]])
      processors[pname] = p
   
   for pname in Yregressors:
      p = preprocessing.RobustScaler()
      p.fit(data[[pname]])
      processors[pname] = p
      
   for pname in Yclassifiers:
      p = preprocessing.OneHotEncoder()
      p.fit(data[[pname]])
      processors[pname] = p
   
   return processors

def safe_model(model, processors, history):
   
   for name, processor in processors.items():
      
      saved_processors = {}
      
      if processor.__class__ == preprocessing.OneHotEncoder:
         p = dict( preprocessor = 'OneHotEncoder', 
                   kwargs = {'categories': processor.categories_, 'drop': processor.drop_idx_})
         
      elif processor.__class__ == preprocessing.StandardScaler:
         p = dict( preprocessor = 'StandardScaler', 
                   kwargs = {'scale_': processor.scale_, 'mean_': processor.mean_, 'var_':processor.var_})
         
      elif processor.__class__ == preprocessing.RobustScaler:
         p = dict( preprocessor = 'RobustScaler', 
                   kwargs = {'scale_': processor.scale_, 'center_': processor.center_})
         
      else:
         p = None
      
      saved_processors[name] = p
   
   
   model_yaml = model.to_yaml()
   
   model_weights = model.get_weights()
   
   #model_json = model.to_json()
   #with open("complete_model.json", "w") as json_file:
      #json_file.write(model_json)
   
   #model.save_weights("complete_model_weights.h5")
   
   
   
def fit_model(setup):
   
   data = pd.read_csv(setup['datafile'])

   Xpars = list(setup['input'])
   Yregressors = list(setup['regressors'])
   Yclassifiers = list(setup['classifiers'])
   
   #Only select the columns that are required.
   data = data[Xpars + Yregressors + Yclassifiers]
   
   #Make the preprocessors
   processors = make_preprocessors(data, Xpars, Yregressors, Yclassifiers)
   
   
   X = np.array([processors[x].transform(data[[x]]) for x in Xpars])
   Y = [processors[x].transform(data[[x]]) for x in Yregressors + Yclassifiers]

   X = X.reshape(X.shape[:-1]).T
   
   model = make_model(X.shape[1], Yregressors, Yclassifiers, processors)

   #from keras.utils import plot_model
   #plot_model(model, to_file='model.png')

   print (model.summary())


   loss = ['mean_squared_error' for name in Yregressors] + ['categorical_crossentropy' for name in Yclassifiers]

   model.compile(optimizer='adam', loss=loss, metrics = ['accuracy', 'mae'])
   history = model.fit(X, Y, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)
   
   fileio.safe_model(model, processors, history, 'test.h5')
   
   model, processors, history = fileio.load_model('test.h5')
   

#-- load the setup file
setupfile = open('bps_setup.yaml')
setup = yaml.safe_load(setupfile)
setupfile.close()

fit_model(setup)





