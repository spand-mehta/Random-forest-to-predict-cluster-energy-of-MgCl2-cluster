# In[6]:
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import os

files=[]
files1=[]
dir_path = os.path.dirname(os.path.realpath(__file__))

path=dir_path+'/parameters/'
path1=dir_path+'/train_energies/'

for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

for r, d, f in os.walk(path1):
        for file in f:
            if '.csv' in file:
                files1.append(os.path.join(r, file))


files=np.sort(files)
files1= np.sort(files1)
fileslist=[]
files1list=[]
for i in files:
	df=pd.read_csv(i)
	X1=df.drop('Unnamed: 0',axis=1)
	fileslist+=[X1]
for i in files1:
	df1=pd.read_csv(i)
	X2=df1.drop('Unnamed: 0',axis=1)
	files1list+=[X2]

X=pd.concat(fileslist)
Yf=pd.concat(files1list)

X.to_csv("verificationX.csv")
Yf.to_csv("verificationY.csv")	

Y=np.ravel(Yf)

import sklearn.model_selection as model_selection
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=45)

regr=RandomForestRegressor(n_estimators=50,max_depth=13)

regr.fit(X,Y)

print X.columns
print regr.feature_importances_

import cPickle

with open('model.cpickle', 'wb') as f:
	cPickle.dump(regr, f)

print "model_created!"