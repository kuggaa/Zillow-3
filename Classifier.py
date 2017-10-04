#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:40:40 2017

@author: saba_rish91
"""
#%%
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from keras import losses
from keras.callbacks import ModelCheckpoint

#%%
##############################################################################################

tData = pd.read_csv('..//Data//train_2016_v2.csv',parse_dates=['transactiondate'])
propData = pd.read_csv('..//Data//properties_2016.csv',low_memory=False)

##############################################################################################

tData['month'] = tData['transactiondate'].dt.month
tData['year'] = tData['transactiondate'].dt.year
tData['quarter'] = tData['transactiondate'].dt.quarter

for col in propData.columns:
    propData[col] = propData[col].fillna(-1)
    if propData[col].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(propData[col].values))
        propData[col] = lbl.transform(list(propData[col].values))

pd.set_option('display.max_columns', None)
combined = tData.merge(propData,how='left',on='parcelid')

del tData
del propData
#%%
########################## Feature extraction #############################################

combined['month1'] = combined['month']**2
combined['month2'] = combined['month']**3

#Feature extraction from Nikunj notebook
#life of property
combined['N-life'] = 2018 - combined['yearbuilt']
#error in calculation of the finished living area of home
combined['N-LivingAreaError'] = combined['calculatedfinishedsquarefeet']/combined['finishedsquarefeet12']
#proportion of living area
combined['N-LivingAreaProp'] = combined['calculatedfinishedsquarefeet']/combined['lotsizesquarefeet']
combined['N-LivingAreaProp2'] = combined['finishedsquarefeet12']/combined['finishedsquarefeet15']

#Amout of extra space
combined['N-ExtraSpace'] = combined['lotsizesquarefeet'] - combined['calculatedfinishedsquarefeet'] 
combined['N-ExtraSpace-2'] = combined['finishedsquarefeet15'] - combined['finishedsquarefeet12'] 
#Total number of rooms
combined['N-TotalRooms'] = combined['bathroomcnt']*combined['bedroomcnt']
#Average room size
#combined['N-AvRoomSize'] = combined['calculatedfinishedsquarefeet']/combined['roomcnt'] 

# Number of Extra rooms
combined['N-ExtraRooms'] = combined['roomcnt'] - combined['N-TotalRooms'] 

#Ratio of the built structure value to land area
combined['N-ValueProp'] = combined['structuretaxvaluedollarcnt']/combined['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
combined['N-GarPoolAC'] = ((combined['garagecarcnt']>0) & (combined['pooltypeid10']>0) & (combined['airconditioningtypeid']!=5))*1 

combined["N-location"] = combined["latitude"] + combined["longitude"]
#combined["N-location-2"] = combined["latitude"]*combined["longitude"]
#combined["N-location-2round"] = combined["N-location-2"].round(-4)

#combined["N-latitude-round"] = combined["latitude"].round(-4)
#combined["N-longitude-round"] = combined["longitude"].round(-4)

#Ratio of tax of property over parcel
combined['N-ValueRatio'] = combined['taxvaluedollarcnt']/combined['taxamount']

#TotalTaxScore
combined['N-TaxScore'] = combined['taxvaluedollarcnt']*combined['taxamount']

#polnomials of tax delinquency year
combined['N-taxdelinquencyyear-2'] = combined['taxdelinquencyyear'] ** 2
combined['N-taxdelinquencyyear-3'] = combined['taxdelinquencyyear'] ** 3

#Number of properties in the zip
zip_count = combined['regionidzip'].value_counts().to_dict()
combined['N-zip_count'] = combined['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = combined['regionidcity'].value_counts().to_dict()
combined['N-city_count'] = combined['regionidcity'].map(city_count)

#Number of properties in the city
#region_count = combined['regionidcounty'].value_counts().to_dict()
#combined['N-county_count'] = combined['regionidcounty'].map(city_count)

#Indicator whether it has AC or not
combined['N-ACInd'] = (combined['airconditioningtypeid']!=5)*1

#Indicator whether it has Heating or not 
combined['N-HeatInd'] = (combined['heatingorsystemtypeid']!=13)*1

#There's 25 different property uses - let's compress them down to 4 categories
combined['N-PropType'] = combined.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })

#polnomials of the variable
combined["structuretaxvaluedollarcnt"] = combined["structuretaxvaluedollarcnt"]/np.max(combined["structuretaxvaluedollarcnt"])
combined["N-structuretaxvaluedollarcnt-2"] = combined["structuretaxvaluedollarcnt"] ** 2
combined["N-structuretaxvaluedollarcnt-3"] = combined["structuretaxvaluedollarcnt"] ** 3

#Average structuretaxvaluedollarcnt by city
group = combined.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
combined['N-Avg-structuretaxvaluedollarcnt'] = combined['regionidcity'].map(group)

#Deviation away from average
combined['N-Dev-structuretaxvaluedollarcnt'] = abs((combined['structuretaxvaluedollarcnt'] - combined['N-Avg-structuretaxvaluedollarcnt']))/combined['N-Avg-structuretaxvaluedollarcnt']

combined = combined.drop(['parcelid','propertyzoningdesc','propertycountylandusecode','fireplaceflag'],axis=1)

#%%
##############################################################################################

yTrain = combined['logerror']
xTrain = combined.drop(['transactiondate','logerror'],axis=1)

del combined

#%%
##############################################################################################

def Model(input_len):
    model = Sequential()
    model.add(Dense(500,input_dim=input_len))
    model.add(PReLU())
    model.add(Dense(250))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

#%%
##############################################################################################
for c in xTrain.dtypes[xTrain.dtypes == object].index.values:
    xTrain[c] = (xTrain[c] == True)
    
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
del group
del col
del city_count
del zip_count

gc.collect()
#%%
##############################################################################################

model = Model(np.shape(xTrain)[1])

bestModelCallBack = ModelCheckpoint(filepath='weights.hdf5',verbose=1,save_best_only=True)

model.compile(loss=losses.mean_squared_error,optimizer=Adam(lr=4e-3,decay=1e-4))
model.fit(np.array(xTrain),np.array(yTrain),batch_size=32,epochs=50,verbose=1,validation_split=0.2)
    
model.load_weights('weights.hdf5')

#%%
