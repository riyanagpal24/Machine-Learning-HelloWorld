# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print(dataset.shape)
print(dataset.head(20))
# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())


# Univariate plots to better understand each attribute

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# hitograms
dataset.hist()
plt.show()


#multivariate plots to better understand the interaction between the variables

# scatter plot matrix
scatter_matrix(dataset)
plt.show()


print("--"*44)