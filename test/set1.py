# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:57:49 2018

@author: riya.nagpal
"""
'''
     to analyse the reviews and predict whether it is recommmended
     by the consumer or not.
 
 '''
from sklearn import datasets
import pandas as pd

data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

Y = data.Recommended_IND
X = data.Rating

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
print(X_train.head())
print(X_train.shape())