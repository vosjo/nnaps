 
import pandas as pd
import numpy as np

import yaml
 
from sklearn import preprocessing

if False:

   setupfile = open('bps_setup.yaml')
   setup = yaml.safe_load(setupfile)
   setupfile.close()


   data = pd.read_csv(setup['datafile'])

   data = data[data['product'] != 'failed']
   
   data['mdot_type'] = list(np.where(data['product'] == 'CE', 'CE', 'stable'))

   Xpars = list(setup['input'])
   Yregressors = list(setup['regressors'])
   Yclassifiers = list(setup['classifiers']) + ['mdot_type', 'delta_max']

   #Only select the columns that are required.
   data = data[Xpars + Yregressors + Yclassifiers]

   for p in Yregressors + Yclassifiers:
      data[p+'_diff'] = np.zeros_like(data[Xpars[0]].values)
      
   print ( data.head() )

   #data['differential_r'] = np.zeros_like(data[Xpars[0]].values)
   #data['differential_c'] = np.zeros_like(data[Xpars[0]].values)

   # scale all parameters
   data_scaled = {}
   for col in Xpars:
      p = preprocessing.StandardScaler()
      data_scaled[col] = p.fit_transform(data[[col]])[:,0]
      data_scaled[col] = data[col]
   for col in Yregressors:
      p = preprocessing.RobustScaler()
      data_scaled[col] = p.fit_transform(data[[col]])[:,0]
   for col in Yclassifiers:
      p = preprocessing.OrdinalEncoder()
      data_scaled[col] = p.fit_transform(data[[col]])[:,0]

   data_scaled = pd.DataFrame(data=data_scaled)

   #print (data_scaled.head())



   for i, line in data_scaled.iterrows():
      
      data_scaled['distance'] = np.sqrt(np.sum((data_scaled[Xpars].values - line[Xpars].values)**2, axis=1))
      
      data_selected = data_scaled.sort_values(by='distance').iloc[1:6]
      
      #print (data_selected)
      
      for p in Yregressors:
         diff = np.average( np.sqrt( (data_selected[p].values - line[p])**2) / data_selected['distance'] )
         
         #data[p+'_diff'][i] = diff
         data.loc[i, p+'_diff'] = diff
         
      #print ( i, diff )
         
      for p in Yclassifiers:
         diff = np.average( np.clip( np.abs( (data_selected[p].values - line[p]) ), 0,1) / data_selected['distance'] )
         
         #data[p+'_diff'][i] = diff
         data.loc[i, p+'_diff'] = diff
      
      #differential_r = np.average( np.sqrt(np.sum((data_selected[Yregressors].values - line[Yregressors].values)**2, axis=1)) / data_selected['distance'] )
      
      #differential_c = np.avera   ge( np.sum( np.clip( np.abs( (data_selected[Yclassifiers].values - line[Yclassifiers].values) ), 0,1), axis=1 ) / data_selected['distance'] )
      
      #print (differential_c)
      
      #data['differential_r'][i] = differential_r
      #data['differential_c'][i] = differential_c
      
      #exit()

   data.to_csv('test_differential.csv')




import pylab as pl

vmax = None
par = 'mdot_type_diff'
reduce_C_function = np.mean

data = pd.read_csv('test_differential.csv')

print (data.head())

#pl.figure(1, figsize=(16, 6))

#ax = pl.subplot(131)

#hb = pl.hexbin(data['M1'], data['Pinit'], data[par], gridsize=30, vmax=vmax, reduce_C_function=reduce_C_function)
#pl.colorbar(hb)


#ax = pl.subplot(132)

#hb = pl.hexbin(data['M1'], data['qinit'], data[par], gridsize=30, vmax=vmax, reduce_C_function=reduce_C_function)
#pl.colorbar(hb)


#ax = pl.subplot(133)

#hb = pl.hexbin(data['M1'], data['FeHinit'], data[par], gridsize=30, vmax=vmax, reduce_C_function=reduce_C_function)
#pl.colorbar(hb)

#pl.tight_layout()


#pl.scatter(data['M1'], data['qinit'], c=list(np.where(data['mdot_type'] == 'CE', 0, 1)) )


par = 'mdot_type_diff'
#par = 'product_diff'
#par= 'Pfinal_diff'
#par = 'delta_max'

vmax = None

d = data.sort_values(by=par)

pl.figure(2, figsize=(16, 6))


ax = pl.subplot(131)
pl.scatter(d['M1'], d['Pinit'], c=d[par], vmax=vmax )

ax = pl.subplot(132)
pl.scatter(d['M1'], d['qinit'], c=d[par], vmax=vmax )

ax = pl.subplot(133)
sc = pl.scatter(d['M1'], d['FeHinit'], c=d[par], vmax=vmax )
pl.colorbar(sc)

pl.tight_layout()


pl.show()
