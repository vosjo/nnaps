
from nnaps import predictors

predictor = predictors.BPS_predictor(setup_file='test_setup2.yaml')

predictor.train(epochs=20, batch_size=128, validation_split=0.20)

predictor.train(epochs=20, batch_size=128, validation_split=0.20)

predictor.train(epochs=20, batch_size=128, validation_split=0.20)

predictor.save_training_history('test_history.csv')

predictor.save_model('test_model.h5')


from nnaps import plotting

import pandas as pd
from sklearn import preprocessing

history = pd.read_csv('test_history.csv')

plotting.plot_training_history_html(history, targets=['Pfinal', 'qfinal', 'product', 'binary_type'], filename='test_training_history.html')

#
#
# training_data = pd.read_csv('BesanconGalactic_summary.txt')
#
# Xpars = ['M1', 'qinit', 'Pinit', 'FeHinit']
# regressors = ['Pfinal', 'qfinal']
#
# processors = dict(M1 = preprocessing.StandardScaler,
#                   qinit = preprocessing.RobustScaler,
#                   Pinit = preprocessing.MinMaxScaler,
#                   FeHinit = preprocessing.MaxAbsScaler,
#                   Pfinal = preprocessing.RobustScaler,
#                 )
#
#
# plotting.plot_training_data_html(training_data, Xpars, regressors, [], processors, filename='/home/joris/Python/nnaps/nnaps/tests/test_training_data.html')