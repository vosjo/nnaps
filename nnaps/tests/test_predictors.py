
import os

import pandas as pd

import  unittest

from sklearn import preprocessing

from nnaps import predictors

from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_path = Path(__file__).parent

class TestBPSPredictor(unittest.TestCase):
   
   def test_make_from_setup(self):
      
      predictor = predictors.BPS_predictor(setup_file=base_path / 'test_setup.yaml')
      Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
      Yregressors_ = ['Pfinal', 'qfinal']
      Yclassifiers_ = ['product', 'binary_type']
      
      # test reading the X and Y variables
      self.assertTrue( all( [a in predictor.Xpars for a in Xpars_] ) )
      self.assertTrue( all( [a in predictor.Yregressors for a in Yregressors_] ) )
      self.assertTrue( all( [a in predictor.Yclassifiers for a in Yclassifiers_] ) )
      
      # test making the preprocessors
      pp = predictor.processors
      
      for par in Xpars_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.StandardScaler, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.StandardScaler, pp[par].__class__))
         
      for par in Yregressors_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.RobustScaler, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.RobustScaler, pp[par].__class__))
         
      for par in Yclassifiers_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.OneHotEncoder, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.OneHotEncoder, pp[par].__class__))
      
      # test making the model
      mod = predictor.model
      
      self.assertEqual(len(mod.layers), 11, 
                       msg="Model does not have correct number of layers, expected {}, got {}".format(11, len(mod.layers)))
      
      self.assertEqual(mod.input.shape[1], len(Xpars_), msg="Model input does not have correct shape, expected {}, got {}".format(len(Xpars_), mod.input.shape[1]))
      
      self.assertEqual(len(mod.output), len(Yregressors_) + len(Yclassifiers_), 
                       msg="Model does not have correct number of outputs, expected {}, got {}".format(len(Yregressors_) + len(Yclassifiers_), len(mod.output)))

   
   def test_make_from_saved_model(self):
      
      predictor = predictors.BPS_predictor(saved_model=base_path / 'test_model.h5')
      
      Xpars_ = ['M1', 'qinit', 'Pinit', 'FeHinit']
      Yregressors_ = ['Pfinal', 'qfinal']
      Yclassifiers_ = ['product', 'binary_type']
      
      # test reading the X and Y variables
      self.assertTrue( all( [a in predictor.Xpars for a in Xpars_] ) )
      self.assertTrue( all( [a in predictor.Yregressors for a in Yregressors_] ) )
      self.assertTrue( all( [a in predictor.Yclassifiers for a in Yclassifiers_] ) )
      
      # test making the preprocessors
      pp = predictor.processors
      
      for par in Xpars_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.StandardScaler, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.StandardScaler, pp[par].__class__))
         
      for par in Yregressors_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.RobustScaler, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.RobustScaler, pp[par].__class__))
         
      for par in Yclassifiers_:
         self.assertTrue(par in pp, msg="{} does not have a preprocessor".format(par))
         self.assertTrue(pp[par].__class__ ==  preprocessing.OneHotEncoder, 
                         msg="{} does not have the correct preprocessor. expected {}, got {}".format(par, preprocessing.OneHotEncoder, pp[par].__class__))
      
      # test making the model
      mod = predictor.model
      
      self.assertEqual(len(mod.layers), 11, 
                       msg="Model does not have correct number of layers, expected {}, got {}".format(11, len(mod.layers)))
      
      self.assertEqual(mod.input.shape[1], len(Xpars_), msg="Model input does not have correct shape, expected {}, got {}".format(len(Xpars_), mod.input.shape[1]))
      
      self.assertEqual(len(mod.output), len(Yregressors_) + len(Yclassifiers_), 
                       msg="Model does not have correct number of outputs, expected {}, got {}".format(len(Yregressors_) + len(Yclassifiers_), len(mod.output)))
      
   
   def test_append_to_history(self):
      
      predictor = predictors.BPS_predictor()
      
      predictor.Yregressors = ['M1final']
      predictor.Yclassifiers = []
      
      data = {'M1final_mae': [0.3, 0.2], 'val_M1final_mae': [0.31, 0.21], 'M1final_loss': [1.5, 1.3], 'val_M1final_loss': [1.6, 1.4], 'training_run': [1, 1]}
      history1 = pd.DataFrame(data=data)
      history1.index.name = 'epoch'
      
      predictor.history = history1
      
      data = {'M1final_mae': [0.1, 0.0], 'val_M1final_mae': [0.11, 0.01], 'M1final_loss': [1.2, 1.1], 'val_M1final_loss': [1.3, 1.2]}
      history2 = pd.DataFrame(data=data)
      history2.index.name = 'epoch'
      
      data = {'M1final_mae': [0.3, 0.2, 0.1, 0.0], 'val_M1final_mae': [0.31, 0.21, 0.11, 0.01], 'M1final_loss': [1.5, 1.3, 1.2, 1.1], 'val_M1final_loss': [1.6, 1.4, 1.3, 1.2], 'training_run': [1, 1, 2, 2]}
      history_expected = pd.DataFrame(data=data)
      history_expected.index.name = 'epoch'
      
      predictor._append_to_history(history2)
      
      history = predictor.history
      
      self.assertTrue(history.equals(history_expected), msg="\nExpected dataframe: \n{}\nGot dataframe: \n {}".format(history_expected.to_string(), history.to_string()) )
   
   
   #def test_train_model(self):
      
      #predictor = predictors.BPS_predictor(setup_file='tests/test_setup.yaml')
      
      #predictor.data = predictor.data.iloc[0:200]
      
      #predictor.train(epochs=2, batch_size=50, validation_split=0.25)
      
      #predictor.train(epochs=2, batch_size=50, validation_split=0.25)
      
      #print (predictor.history)
      
      #self.assertTrue(False)


   #def test_predict(self):
      
      #data = pd.read_csv('tests/BesanconGalactic_summary.txt').iloc[0:10]
      
      #predictor = predictors.BPS_predictor(saved_model='tests/test_model.h5')
      
      #res = predictor.predict(data=data)
      
      #print (res)
      
      #self.assertTrue(False)
      
