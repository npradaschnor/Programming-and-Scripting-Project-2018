#Noa P Prada Schnor 2018-03-24
#Based on #https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#Based on https://campus.datacamp.com/courses/pandas-foundations/data-ingestion-inspection?ex=1
#Based on https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

#import key libraries to help to analyse the data set
import csv
import pandas as pd
import scipy
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright') #style of plot = seaborn-bright (I can change based on what is available from print plt.style.available)
import sklearn
from sklearn import datasets
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
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.decomposition import PCA
import scikit-learn

print(plt.style.available)  # list all available styles (matplot)

with open ('data/iris.csv') as f: #open iris data set
  names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
  dataset = pandas.read_csv(f, names=names) #give each column a 'header' a 'name'
  
  print(dataset.head()) #to check the header
  
  #info about the dataset
  print(type(dataset)) #type
  print(dataset.shape) # get the info about row many rows and columns
  print(dataset.columns) #name of the columns
  print(type(dataset.columns))#pandas index
  print(dataset.index) #daytime index
  print(dataset.tail(20)) #returns the last 20 rows, so I can see how my data looks like.. I could use the dataset.head(20) to check the first 20 rows
  print(dataset.info()) #returns an index: datatimeindex, number of columns, type of data in each column, data types of the whole dataset, etc.

  # describe the data: count, mean, minimun/maximum values and percentiles.
  print(dataset.describe())
  
  dataset.hist('sepal-length') #plot the histogram of sepal lenght
  plt.title('Histogram of sepal lenght') #title of histogram
  plt.xlabel('Sepal lenght in cm') #x axis label
  plt.ylabel('Number of sample') #y axis label
  plt.savefig('iris_hist_sepallenght.png') #save plot
  plt.show() #show the histogram

  dataset.hist('sepal-width')  # plot the histogram of sepal width
  plt.title('Histogram of sepal width') #title of histogram
  plt.xlabel('Sepal width in cm')  # x axis label
  plt.ylabel('Number of sample')  # y axis label
  plt.savefig('iris_hist_sepalwidth.png') #save plot
  plt.show()  # show the histogram

  dataset.hist('petal-length')  # plot the histogram of petal lenght
  plt.title('Histogram of petal lenght') #title of histogram
  plt.xlabel('Petal lenght in cm')  # x axis label
  plt.ylabel('Number of sample')  # y axis label
  plt.savefig('iris_hist_petallenght.png') #save plot
  plt.show()  # show the histogram

  dataset.hist('petal-width')  # plot the histogram of petal width
  plt.title('Histogram of petal width') #title of histogram
  plt.xlabel('Petal width in cm')  # x axis label
  plt.ylabel('Number of sample')  # y axis label
  plt.savefig('iris_hist_petalwidth.png') #save plot
  plt.show()  # show the histogram
  sl_arr = dataset['sepal-length'].values
  print(type(sl_arr)) #print the type of var

  plt.plot(sl_arr) #plot array (matplotlib) sepal lenght
  plt.savefig('iris_plotarray_sepallenght.png') #save plot
  plt.show() # show the plot

  dataset.plot() #plot dataframe (pandas)
  plt.savefig('iris_plotdataframe.png') #save plot
  plt.show() # show the plot

  dataset.plot()
  plt.yscale('log') #fixing scales - log scale on vertical axis
  plt.savefig('iris.log_verticalaxis.png') #save plot
  plt.show() # show the plot

  dataset.plot(kind='barh', stacked=True)#multiple bar plot
  plt.savefig('iris_plotdataframe.png')
  plt.xlabel('in cm')  # x axis label
  plt.ylabel('Sample of 150 flowers')  # y axis label
  plt.show() #show the plot
  
  print(pd.isnull(dataset)) # To identify the rows that contain missing values. True will indicate that the value contained within the cell is a missing value, False means that the cell contains a ‘normal’ value. In this case, there are no missing values.

dataset = datasets.load_iris() #another way of 	Load and return the iris dataset (classification) via sklearn

X = dataset.data
y = dataset.target #The target attribute is the integer index of the category
tgn = dataset.target_names #"label names"

pca = PCA(n_components=2) #Number of components to keep
X_r = pca.fit(X).transform(X) #Fit the model with X and apply the dimensionality reduction on X

lda = LinearDiscriminantAnalysis(n_components=2) #Number of components to keep
X_r2 = lda.fit(X, y).transform(X) #Fit the model with X and apply the dimensionality reduction on X

print(pca.explained_variance_ratio_)#The amount of variance explained by each of the selected components

plt.figure()
colors = ['magenta', 'green', 'blue'] #Colors chose to plot ("dots" color)
lw = 2 #linewidth

for color, i, tgn in zip(colors, [0, 1, 2], tgn): 
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=tgn) #X,Y = data positions; Alpha = blending value, between 0 (transparent) and 1 (opaque); lw = the linewidth of the marker edges. Note: The default edgecolors is ‘face’. You may want to change this as well. If None, defaults to rcParams lines.linewidth.The linewidth of the marker edges. The default edgecolors is ‘face’.
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Principal Component Analysis (PCA) of Iris dataset')#plot's title
plt.savefig('iris_PCA.png')#save the fig named "iris-PCA.png"

plt.figure()
for color, i, tgn in zip(colors, [0, 1, 2], tgn):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=tgn) #X,Y = data positions; Alpha = blending value, between 0 (transparent) and 1 (opaque); lw = the linewidth of the marker edges. Note: The default edgecolors is ‘face’. You may want to change this as well. If None, defaults to rcParams lines.linewidth.The linewidth of the marker edges.
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Linear Discriminant Analysis (LDA) of Iris dataset')#plot's title
plt.savefig('iris_LDA.png')#save the fig named "iris-LDA.png"

plt.show() #show the plots
  
