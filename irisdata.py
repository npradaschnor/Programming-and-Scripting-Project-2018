#Noa P Prada Schnor From 2018-03-24 to 2018-04-29
# Python script to analyse Iris Dataset

#IMPORTING KEY LIBRARIES TO HELP TO ANALYSE THE DATASET
import csv
import pandas
import scipy
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# style of plot = seaborn-bright (I can change based on what is available from print plt.style.available)
plt.style.use('seaborn-bright')
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
from sklearn import model_selection
from sklearn.decomposition import PCA

print(plt.style.available)  # list all available styles (matplot)

# 1 OPEN IRIS DATASET
with open('data/iris.csv') as f:  # open iris data set
  names = ['sepal-length', 'sepal-width',
           'petal-length', 'petal-width', 'class']
  # give each column a 'header' a 'name'
  dataset = pandas.read_csv(f, names=names)

## 1.1 OTHER WAYS OF OPENING IRIS DATASET
dataset1 = datasets.load_iris() #via sklearn

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names1 = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset2 = pandas.read_csv(url, names=names) #via Panda using URL

# 2 CHECKING THE IRIS DATASET - BASIC INFO BEFORE ANALYSING IT

## 2.1 CHECKING MISSING VALUES
 
# To identify the rows that contain missing values. True will indicate that the value contained within the cell is a missing value, False means that the cell contains a ‘normal’ value. In this case, there are no missing values.
print(pandas.isnull(dataset))

## 2.2 TYPE OF DATASET
print(type(dataset))

## 2.3 NUMBER OF ROWS AND COLUMNS
print(dataset.shape) 
  
## 2.4 NAME OF THE COLUMNS
print(dataset.columns)  # name of the columns
  
## 2.5 CLASS DISTRIBUTION
print(dataset.groupby('class').size())

## 2.6 DATASET INDEX
print(dataset.info()) #index:datatimeindex,n.of columns,type of data of each column, data types of the whole dataset, etc.

## 2.7 HOW THE DATA LOOKS LIKE
print(dataset.tail(20)) #returns the last 20 rows. I could check the first 20 rows: dataset.head(20)

## 2.8 PIVOT TABLE
print(dataset.pivot_table(index='class', values=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'], aggfunc=np.mean))#pivot table with their means

## 2.9 DATA DESCRIPTION
print(dataset.describe())  # count, mean, minimun/maximum values and percentiles.

# 3 PLOTS

## 3.1 HISTOGRAMS

###3.1.1 HISTOGRAM OF EACH ATTRIBUTE

dataset.hist('sepal-length')  # plot the histogram of sepal lenght
plt.title('Histogram of sepal lenght')  # title of histogram
plt.xlabel('Sepal lenght in cm')  # x axis label
plt.ylabel('Number of sample')  # y axis label
plt.savefig('iris_hist_sepallenght.png')  # save plot
plt.show()  # show the histogram

dataset.hist('sepal-width')  # plot the histogram of sepal width
plt.title('Histogram of sepal width')  # title of histogram
plt.xlabel('Sepal width in cm')  # x axis label
plt.ylabel('Number of sample')  # y axis label
plt.savefig('iris_hist_sepalwidth.png')  # save plot
plt.show()  # show the histogram

dataset.hist('petal-length')  # plot the histogram of petal lenght
plt.title('Histogram of petal lenght')  # title of histogram
plt.xlabel('Petal lenght in cm')  # x axis label
plt.ylabel('Number of sample')  # y axis label
plt.savefig('iris_hist_petallenght.png')  # save plot
plt.show()  # show the histogram

dataset.hist('petal-width')  # plot the histogram of petal width
plt.title('Histogram of petal width')  # title of histogram
plt.xlabel('Petal width in cm')  # x axis label
plt.ylabel('Number of sample')  # y axis label
plt.savefig('iris_hist_petalwidth.png')  # save plot
plt.show()  # show the histogram

###3.1.2 HISTOGRAM OF ALL 4 ATTRIBUTES
dataset.hist() #histogram plot of all 4 attributes
plt.savefig('iris_hist.png')  # save plot
plt.show() #show plot

## 3.2 PLOT DATAFRAME/SERIES
 
### 3.2.1 SEPAL LENGHT
sl_arr = dataset['sepal-length'].values
print(type(sl_arr))  # print the type of var
  
plt.plot(sl_arr)  # plot array (matplotlib) sepal lenght
plt.savefig('iris_plotarray_sepallenght.png')  # save plot
plt.show()  # show the plot

# 3.2.2 ALL 4 ATTRIBUTES
dataset.plot()  # plot dataframe (pandas)
plt.savefig('iris_plotdataframe.png')  # save plot
plt.show()  # show the plot

# 3.2.3 ALL ATTRIBUTES Y AXIS ON LOG SCALE
dataset.plot()
plt.yscale('log')  # fixing scales - log scale on vertical axis
plt.savefig('iris.log_verticalaxis.png')  # save plot
plt.show()  # show the plot

## 4 MULTIPLE BAR PLOT-STACKED BAR GRAPH
dataset.plot(kind='barh', stacked=True)  # multiple bar plot
plt.savefig('iris_plotdataframe.png') #save the plot
plt.xlabel('in cm')  # x axis label
plt.ylabel('Sample of 150 flowers')  # y axis label
plt.show()  # show the plot

# 5 BOX AND WHISKER PLOT
color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')#colors
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False, color=color)#plot type box
plt.savefig('iris_box_and_whisker_plot.png') #save the plot
plt.show() #show the plot

## 6 SCATTER PLOTS

dataset = datasets.load_iris() #load iris dataset via sklearn

### 6.1 LINEAR TRANSFORMATION TECHNIQUES

X = dataset.data
y = dataset.target  # The target attribute is the integer index of the category
tgn = dataset.target_names  # "label names"

### 6.1.1 PRINCIPAL COMPONENT ANALYSIS - PCA

pca = PCA(n_components=2)  # Number of components to keep
# Fit the model with X and apply the dimensionality reduction on X
X_r = pca.fit(X).transform(X)

### 6.1.2 LINEAR DISCRIMINANT ANALYSIS - LDA

lda = LinearDiscriminantAnalysis(
    n_components=2)  # Number of components to keep
# Fit the model with X and apply the dimensionality reduction on X
X_r2 = lda.fit(X, y).transform(X)

plt.figure()
colors = ['magenta', 'green', 'blue']  # Colors chose to plot ("dots" color)
lw = 2  # linewidth

for color, i, tgn in zip(colors, [0, 1, 2], tgn):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=tgn)  # X,Y = data positions; Alpha = blending value, between 0 (transparent) and 1 (opaque); lw = the linewidth of the marker edges. Note: The default edgecolors is ‘face’. You may want to change this as well. If None, defaults to rcParams lines.linewidth.The linewidth of the marker edges. The default edgecolors is ‘face’.
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Principal Component Analysis (PCA) of Iris dataset')  # plot's title
plt.savefig('iris_PCA.png')  # save the fig named "iris-PCA.png"

plt.figure()
for color, i, tgn in zip(colors, [0, 1, 2], tgn):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=tgn)  # X,Y = data positions; Alpha = blending value, between 0 (transparent) and 1 (opaque); lw = the linewidth of the marker edges. Note: The default edgecolors is ‘face’. You may want to change this as well. If None, defaults to rcParams lines.linewidth.The linewidth of the marker edges.
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Linear Discriminant Analysis (LDA) of Iris dataset')  # plot's title
plt.savefig('iris_LDA.png')  # save the fig named "iris-LDA.png"

plt.show()  # show PCA and LDA plots

# 7 VARIANCE RATIO OF THE TWO SELECTED COMPONENTS
# The amount of variance explained by each of the selected components
print(pca.explained_variance_ratio_)
