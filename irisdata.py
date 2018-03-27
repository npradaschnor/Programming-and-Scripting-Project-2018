#Noa P Prada Schnor 2018-03-24
#Based on #https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#Based on https://campus.datacamp.com/courses/pandas-foundations/data-ingestion-inspection?ex=1

#import key libraries to help to analyse the data set
import csv
import pandas
import scipy
import numpy
import sklearn
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright') #style of plot = seaborn-bright (I can change based on what is available from print plt.style.available)
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
from sklearn import datasets

print(plt.style.available)  # list all available styles (matplot)

with open ('data/iris.csv') as f: #open iris data set
  names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
  dataset = pandas.read_csv(f, names=names) #give each column a 'header' a 'name'
  
#get to knowing the dataset
  print(type(dataset)) #type
  print(dataset.shape) # get the info about row many rows and columns
  print(dataset.columns) #name of the columns
  print(type(dataset.columns))#pandas index
  print(dataset.index) #daytime index
  print(dataset.tail(50)) #returns the last 50 rows, so I can see how my data looks like.. I could use the dataset.head(50) to check the first 50 rows
  print(dataset.info()) #returns a index: datatimeindex, number of columns, type of data in each column, data types of the whole dataset, etc.

  # describe the data: count, mean, minimun/maximum values and percentiles.
  print(dataset.describe())

  dataset.hist('sepal-length') #plot the histogram of sepal lenght
  plt.savefig('iris_hist_sepallenght.png') #save plot
  plt.show() #show the histogram

  dataset.hist('sepal-width')  # plot the histogram of sepal width
  plt.savefig('iris_hist_sepalwidth.png') #save plot
  plt.show()  # show the histogram

  dataset.hist('petal-length')  # plot the histogram of petal lenght
  plt.savefig('iris_hist_petallenght.png') #save plot
  plt.show()  # show the histogram

  dataset.hist('petal-width')  # plot the histogram of petal width
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
