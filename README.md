# Programming-and-Scripting-Project-2018

###### Project instruction
The project entails the student researching the data set, and then writing documentation and code in the Python programming language based on that research. An online search for information on the data set will convince the student that many people have investigated and written about it previously, and many of those are not experienced programmers. The student is expected to be able to break this project into several smaller tasks that are easier to solve, and to plug these together after they have been completed. 

###### Smaller tasks that have to be done to complete the project:
- [x] Research background information about the data set and write a summary about it
- [x] Keep a list of references you used in completing the project
- [x] Download the data set and write some Python code to investigate it
- [x] Summarise the data set by, for example, calculating the maximum, minimum and mean of each column of the data set. A Python script       will quickly do this for you
- [x]  Write a summary of your investigations
- [x] Include supporting tables and graphics as you deem necessary

#### Python script is on :file_folder: "irisdata.py"

## Part 1: About Iris Dataset
 [[back to top](#project-instruction)]
 
  Iris dataset is a multivariate dataset of three classes of Irises and it was collected by the American botanist Edgar Anderson (1935) and introduced by the British statistician and geneticist Ronald Fisher in his article published in 1936 <i>"The Use of Multiple Measurements in Taxonomic Problems"<i> introducing linear-discriminant-function technique. Fisher's paper is referenced frequently to this day for being such a classic in the field. The Iris data set is a best known and understood dataset and one of the most used to analyse data sets in statistics, data visualization, machine learning, etc [4, 6].The iris dataset is available online from University California Irvine's (UCI) machine-learning repository of datasets (http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) [1,2,3,5,6]. 
 
<img src="https://image.ibb.co/cbYW47/anderson_edgar_pdf.png" style="width: 700px; height:600px;"> <br>
<img src="https://image.ibb.co/jUYu4x/R_A_Fischer.jpg" style="width: 800px; height:700px;"><br>
Edgar Anderson and Ronald Fisher

   The datasset contains data with 150 random samples of flowers of 50 instances from each of three species of irises (*setosa*, *versicolor* and *virginica*). Two parts of the flower were measured: sepal and pedal (lenght and width of each part in cm, total of four features)[1]. 
   
 <img src="https://image.ibb.co/gbSzP7/irisdataset.png"><br>
 Iris attributes. From: https://rpubs.com/wjholst/322258
  
  Therefore, there are 5 attributes in the data base: sepal lenght in cm, sepal widht in cm, petal lenght in cm, petal width in cm and class - *iris setosa*, *iris versicolor* and *iris virginica* [2,4,6,7]. The archive of Iris dataset can be view with any text editor and it contains 5 columns in each row: first four is the features and the fifth one is the label[1].
 
 <img src="https://image.ibb.co/kTbs2H/2018_04_11_13_05_03_sq_Es_Wbo_png_451_592.png"><br>
 Iris Dataset characteristics. From: http://scikit-learn.org/stable/datasets/index.html#datasets

#### References (Part 1)
[[back to top](#project-instruction)]

<ol type="[1]">
  <li> Predictive Analytics For Dummies Available: http://www.dummies.com/DummiesTitle/Predictive-Analytics-For-Dummies.productCd-1118728963,navId-322439,descCd-DOWNLOAD.html</li>
  <li>https://www.lynda.com/Apache-Spark-tutorials/Preprocessing-Iris-data-set/559180/674634-4.html</li>
  <li>http://www.fon.hum.uva.nl/praat/manual/iris_data_set.html</li>
  <li>http://www.idvbook.com/teaching-aid/data-sets/the-iris-data-set/</li>
  <li>http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf</li>
  <li>https://technichesblog.wordpress.com/2015/10/25/matlab-code-to-import-iris-data/</li>
  <li>https://statistics.laerd.com/statistical-guides/understanding-histograms.php</li>
  </ol>

## Part 2: My investigations (DRAFT)
[[back to top](#project-instruction)]

Before starting the project some programs, files and libraries must be downloaded and installed:
<ol>
 <li>Python version 3.6 downloaded via Anaconda3</li>
 <li>Visual Studio Code version 1.21.1 downloaded and set up with Github</li> 
 <li>Iris dataset downloaded from UCI website and from other ways, such as via sklearn </li>
 <li>Libraries imported: csv, pandas, numpy, matplotlib and sklearn </li>
 </ol>
 
 When a library is imported, it means that the library will be loaded into the memory and then it can be use used. To import a library the following code should be run:
 
 ```
 import csv
 import pandas as pd
 import numpy as np
 import sklearn
 ```
 
About some of the libraries imported[1]: 
_Pandas_ - for data-frame management package that allows for some useful function on the dataset.
_Numpy_ - package useful for lineal algebra.
_Matplotlib_ - good package to contruct visualizations.
_Sklearn_ - package important to do machine learning in python.

I've tried to convert the data from iris data set from strings to float numbers (def function make float) but it didn't work (the last columns couldn't be converted to float or number). 

```
def makefloat(teststr):
 try:
  return float(teststr)  # return a floating point value if possibe.
 except:
  return teststr  # otherwise return the string as is*
 ```   

Every time I tried to deal with dataset I couldn't because the last column contains text. Then, I've tried to delete the last column (named class), so I could get a dataset consisted only with numbers using pandas as it follows.

```
data = pandas.read_csv('data/iris.csv')  # select csv file
data = data.drop(['class'], axis=1)  # delete the column named 'class'
data.to_csv('data/datairis.csv') # create new file (that not contains the column 'class')*
```

After I got a dataset that contained only numbers I tried to find a way to find the min and max values of each column. But every code that I've tried didn't work.

```
with open("data/datairis.csv", "r") as f:
  
  lmin_row = []
  lmin_col = []
  
  for row in csv.reader(f):
   row = map(float, row)
   lmin_row.append(min(row))

   if lmin_col:
    lmin_col = map(min, lmin_col, row)
   
   else:
    lmin_col = row
  print("Min per row:", lmin_row)
  print("Min per col:", lmin_col)
  
 -> lmin_row.append(min(row)) ValueError: could not convert string to float: 'sepal length'*
  ```
 
:clipboard: Note: the attempts to deal with Iris Dataset that didn't have sucess are not shown in the code block of 'irisdata.py' file.

[[back to top](#project-instruction)]

After so many attempts to get the min and max values of each column I've started to read more about the Iris Data Set and I've found an article about Iris Data Set and Machine Learning written by Jason Brownlee, Ph.D and he aimed to help professional developers to get started and confidently apply machine learning to address complex problems. In the article Jason Brownlee explains in a simple way about how to do machine learning using Python and he used Iris Data Set as an example using some key libraries , such as numpy, pandas, matplotlib, scipy and sklearn. In a easy and fast way I could analyse the data: getting descriptions (mean, min, max values and percentiles) of each attribute, creating histogram of each numerical variable. Moreover, I've found an online course about Pandas DataFrame on DataCamp and it help me to understand some code/functions and based on the videos I could create some of the codes, specially to get more info about the data set that I'm working on it. All these websites opened my mind to Pandas, Numpy, Matplot, etc. Then, it got easier to understand how I could get graphics in a few lines of code. Besides, reading some tutorials (Pyplot, Numpy, Pandas) I could improve my plots by giving them a title, axis label, etc.
  
  That way, after I read so many articles about Iris dataset and machine learning, Exploratory Data Analysis the following steps have been done:
  
:arrow_right: Print all styles available (Matplot)
  
 :heavy_check_mark: Line 33 on :file_folder: irisdata.py
  
  ```
  print(plt.style.available)
  ```
  
 <img src="https://image.ibb.co/kCGz1n/irisdata_styles_available_matplot.png" alt="irisdata_styles_available_matplot" border="0">
 
 Then, I ran simple descriptive statistics on iris data set. That way, I could effectively appproach the iris data set.
      
:arrow_right: # 1 Open the Iris dataset and each column got a 'header'
  
 :heavy_check_mark: Lines 36 - 40 on :file_folder: irisdata.py
  ```
with open ('data/iris.csv') as f: #open iris data set
   names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
   dataset = pandas.read_csv(f, names=names)</b>
   ```
  
:arrow_right: #2 Information about the dataset: type, number of rows and columns, name of the columns, pandas index, datatimeindex(number of columns, type of data in each column, ...). Moreover, to get a general idea about the dataset and to take a closer look at the data itself the function  (dataset.tail(20)) was used to check out the last 20 rows. That way, I got a general idea about the data set.

##2.1 Checking missing values
To identify the rows that contain missing values. True will indicate that the value contained within the cell is a missing value, False means that the cell contains a ‘normal’ value. In this case, there are no missing values.

:heavy_check_mark: Line 54 on :file_folder: irisdata.py
 
   ```
print(pandas.isnull(dataset))
   ```

<img src="https://image.ibb.co/hemGgn/irisdata_isnull.png" alt="irisdata_isnull" border="0">
   
 ##2.2 Type of dataset
  
 :heavy_check_mark: Line 57 on :file_folder: irisdata.py
  
  ```
print(type(dataset)) 
  ```

<img src="https://image.ibb.co/mNM4Z7/irisdata_data_type.png" alt="irisdata_data_type" border="0">

 ##2.3 Number of rows and columns
  
 :heavy_check_mark: Line 60 on :file_folder: irisdata.py
  
  ```
print(dataset.shape)
 ``` 
 
 <img src="https://image.ibb.co/eR9rE7/irisdata_number_of_rows_and_columns.png" alt="irisdata_number_of_rows_and_columns" border="0">
 
 ##2.4 Name of the columns
 
 :heavy_check_mark: Line 63 on :file_folder: irisdata.py
 
  ```
print(dataset.columns)
  ```

<img src="https://image.ibb.co/gNWou7/irisdata_index_header.png" alt="irisdata_index_header" border="0">

 ##2.5 Class distribution. There are 150 samples of flowers: 50 are <i>iris setosa<i>, 50 are <i>iris versicolor<i> and 50 are <i>iris virginica<i>
 
 :heavy_check_mark: Line 66 on :file_folder: irisdata.py
 
  ```  
print(dataset.groupby('class').size())
   ```
  
<img src="https://image.ibb.co/ngabgn/irisdata_class.png" alt="irisdata_class" border="0">
   
 ##2.6 Dataset index: datatimeindex,n.of columns,type of data of each column, data types of the whole dataset, etc.

 :heavy_check_mark: Line 69 on :file_folder: irisdata.py
   
   ```
print(dataset.info()) 
  ```
  
 <img src="https://image.ibb.co/crFv7S/irisdata_datetime.png" alt="irisdata_datetime" border="0">
  
 ##2.7 Checking the last 20 rows of the dataset.
  
 :heavy_check_mark: Line 72 on :file_folder: irisdata.py
  
  ```
print(dataset.tail(20)) 
  ```

<img src="https://image.ibb.co/k0XsnS/irisdata_20_rows_tail.png" alt="irisdata_20_rows_tail" border="0">

##2.8 Pivot table: tool used to organize and summarize data between databases. It facilitates rotational, pivotal and/or structural changes[2].

:heavy_check_mark: Line 75 on :file_folder: irisdata.py

 ```
print(dataset.pivot_table(index='class', values=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'], aggfunc=np.mean))
 ```
 
 <img src="https://image.ibb.co/k0XsnS/irisdata_20_rows_tail.png" alt="irisdata_20_rows_tail" border="0">
 
##2.9 Data description. Summary statistics that exclude NaN values -  table that summarises the numeric information in the dataset, such as count, mean, standard deviation, minimum and maximum values and the quantiles of the data. It  was useful to gain a general sense of how the data is structured and about the data per se. 

:heavy_check_mark: Line 78 on :file_folder: irisdata.py

  ```
print(dataset.describe())
   ```
   
<img src="https://image.ibb.co/ccvj1n/irisdata_describe.png" alt="irisdata_describe" border="0">
   
:arrow_right:#3 Plots

[[back to top](#project-instruction)]

:arrow_right_hook: ##3.1 Histograms: 'A histogram is a plot that lets you discover, and show, the underlying frequency distribution (shape) of a set of continuous data. This allows the inspection of the data for its underlying distribution (e.g., normal distribution), outliers, skewness, etc'[4,5].

###3.1.1 Histogram of each attribute. It was created separate histogram for each attribute.

:heavy_check_mark: Lines 86 - 112 on :file_folder: irisdata.py
 
  ```
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
  ```
  
  <img src="https://image.ibb.co/hy1BUx/iris_hist_sepallenght.png" alt="iris hist sepallenght" border="0" />
  <img src="https://image.ibb.co/cq9y9x/iris_hist_sepalwidth.png">
    <img src="https://image.ibb.co/kjvuhH/iris_hist_petallenght.png">
    <img src="https://image.ibb.co/jPrBUx/iris_hist_petalwidth.png">

###3.1.2 Histogram of all attributes. It was created one single plot for all attributes.

:heavy_check_mark: Lines 115 - 117 on :file_folder: irisdata.py

```
dataset.hist() #histogram plot of all 4 attributes
plt.savefig('iris_hist.png')  # save plot
plt.show() #show plot
```
<img src="https://image.ibb.co/e54PZ7/iris_hist_allattributes.png">

:arrow_right_hook: ##3.2 Plot on Dataframe/Series: The x axis represents the sample of 150 flowers and the y axis represents the lenght of sepal (in cm) of each flower. 
 
 ###3.2.1 Sepal lenght 
 
:heavy_check_mark: Lines 122 - 127 on :file_folder: irisdata.py
 
  ```
sl_arr = dataset['sepal-length'].values
print(type(sl_arr)) #print the type of var
  
plt.plot(sl_arr) #plot array (matplotlib) sepal lenght
plt.savefig('iris_plotarray_sepallenght.png') #save plot
plt.show() # show the plot
  ```
  
  <img src="https://image.ibb.co/bMqibc/iris_plotarray_sepallenght.png">
 
 ###3.2.2 All attributes

:heavy_check_mark: Lines 130 - 132 on :file_folder: irisdata.py

  ```
dataset.plot()  # plot dataframe (pandas)
plt.savefig('iris_plotdataframe.png')  # save plot
plt.show()  # show the plot
  ```
  
<img src="https://image.ibb.co/m9EzZ7/allatributes.png" alt="allatributes" border="0" />

###3.2.3 All attributes y axis as log-scale.A base-10 log scale is used for the Y axis of the bottom left graph, and the Y axis ranges from 0.1 to 1,000[6].
 
:heavy_check_mark: Lines 135 - 138 on :file_folder: irisdata.py
 
 ```
  dataset.plot()
  plt.yscale('log') #fixing scales - log scale on vertical axis
  plt.savefig('iris.log_verticalaxis.png') #save plot
  plt.show() # show the plot
```

  <img src="https://image.ibb.co/gbEn2H/iris_log_verticalaxis.png">
  
:arrow_right_hook: #4 Multiple bar plot - Stacked Bar Graph: two or more sets of data are represented. It faciliates comparison between more than one phenomena. In this project, a stacked bar graph (one type of multiple bar plot) was plotted. It shows sub-groups that are displayed on the same bar[7]. So, each sample of flower is a bar and each bar contains 4 attributes of the flower (sepal lenght, sepal width, petal lenght and petal width).

:heavy_check_mark: Lines 141 - 145 on :file_folder: irisdata.py
 
 ```
dataset.plot(kind='barh', stacked=True)  # multiple bar plot
plt.savefig('iris_plotdataframe.png') #save the plot
plt.xlabel('in cm')  # x axis label
plt.ylabel('Sample of 150 flowers')  # y axis label
plt.show()  # show the plot
 ```
 
<img src="https://image.ibb.co/j3O0cH/Figure_1.png">

:arrow_right_hook:#5 Box and Whisker Plot:

:heavy_check_mark: Lines 148 - 151 on :file_folder: irisdata.py
 
 ```
color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')#colors
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False, color=color)#plot type box
plt.savefig('iris_box_and_whisker_plot.png') #save the plot
plt.show() #show the plot
 ```

<img src="https://image.ibb.co/c8av7S/iris_box_and_whisker_plot.png" alt="iris_box_and_whisker_plot" border="0">
 
:arrow_right_hook:#6 Scatter plot: A scatter plot is a graph in which a set of points plotted on a horizontal and vertical axes.As scatter plots show the extent of correlation between the values of the variables, they are an important tool in statistics. If there is no correlation between the variables the points will show randomly scattered on the coordinate plane. But, if there is a large correlation between the variables, the points concentrate near a straight line. Therefore, scatter plots are quite useful for data visualization as they illustrate a trend. Scatter plots shows not only the extent of correlation, but also the sense of the correlation. Neverthless, scatter plots not show the causation[8,9,10]. 
  So, if the vertical (called y axis) variable increases and the horizontal (called x axis) variable also increases it means that there is a correlation (positive correlarion). The maximum positive correlation that is possible is +100% or +1 (all points in the plot lie along a straight line in a positive slope). In case the y axis variable decreases and the x axis increases or vice-versa it is a negative correlation. The maximum negative correlation that is possible is -100% or -1 (all the points in the plot lie along a straight line in a negative slope)[8,9,10].  
  
##6.1 Linear transformation techniques - Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA)

:heavy_check_mark: Lines 155 -195 on :file_folder: irisdata.py
 
 ```
 dataset = datasets.load_iris() #load iris dataset via sklearn



X = dataset.data
y = dataset.target  # The target attribute is the integer index of the category
tgn = dataset.target_names  # "label names"



pca = PCA(n_components=2)  # Number of components to keep
# Fit the model with X and apply the dimensionality reduction on X
X_r = pca.fit(X).transform(X)



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
 ```

###6.1.1 PCA: identifies the combination of attributes (principal components, or directions in the feature space) that account for the most variance in the data. It is a linear dimensionality reduction that uses Singular Value Decomposition of the data of dataset to a lower dimensional space. In this case, samples of the two fisrt components were plotted. Usually, PCA is used to speed up a machine learning algorithm, helping to visualize your data. As Iris dataset is 4 dimensional the data must be projects into 2, that way it is easy to understand better the data and it is easier to plot. After the reduction from 4 to 2 dimension data usually there is no particular meaning designated to each main component[11]. 

  <img src="https://image.ibb.co/kRLD9x/PCAIris.png">
  <img src="https://image.ibb.co/dvzfgn/PCAIrisline.png">
    
###6.1.2 LDA: indentifies attributes that account for the most variance between classes. LDA In particular, LDA, in contrast to PCA, is a supervised method, using known class labels[11,13].
 
 <img src="https://image.ibb.co/fqX4Gc/LDAIris.png">
 <img src="https://image.ibb.co/fCP2u7/LDAIrisline.png">
  
 After plotting PCA and LDA it is possible to see relationship between their features of irises, especially <i>iris setosa <i> (in magenta) from those of the others. Another point that can be noticed is that there is some overlapping between <i>iris versicolor<i> (in green) and <i>iris virginica<i> (in blue). That way, one class clearly is linearly separable from the other two (<i>setosa<i> vs <i>versicolor/virginica<i>, but the last two are not totally linealy separable from each other (<i>versicolor<i> vs <i>virginica<i>) .
 
:heavy_exclamation_mark:PCA vs LDA: Both LDA and PCA are used for dimensionality reduction, but PCA is described as an unsupervised method, because it does not take into account class labels and its object is to find the directions, known as principal components that maximize the variance in a dataset. While LDA is supervised method and computes the directions (“linear discriminants”) that will represent the axes that that maximize the separation between multiple classes[11,12,13].

:arrow_right_hook:#7 Variance ratio:

:heavy_check_mark: Line 199 on :file_folder: irisdata.py

 ```
# The amount of variance explained by each of the selected components
print(pca.explained_variance_ratio_)
  ```

[[back to top](#project-instruction)]

#### References (Part 2)
[1]http://www.philipkalinda.com/ds3.html<br>
[2]https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781783553358/4/ch04lvl1sec47/pivot-tables<br>
[3]https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html<br>
[4]https://statistics.laerd.com/statistical-guides/understanding-histograms.php<br>
[5]https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html<br>
[6]https://en.wikipedia.org/wiki/Logarithmic_scale<br>
[7]https://www.emathzone.com/tutorials/basic-statistics/multiple-bar-chart.html#ixzz5CkSyBr12<br>
[8]https://whatis.techtarget.com/definition/scatter-plot<br>
[9]http://mste.illinois.edu/courses/ci330ms/youtsey/scatterinfo.html<br>
[10]http://www.stat.yale.edu/Courses/1997-98/101/scatter.htm<br>
[11]https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60<br>
[12]http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html<br>
[13]http://sebastianraschka.com/Articles/2014_python_lda.html<br>

[[back to top](#project-instruction)]

LEMBRETE PARA MIM MESMA
- ajeitar as images
- verificar se todos os graficos foram colocados no arquivo
- tirar screenshot dos resultados e colocar no arquivo
- descrever o variance ratio, box and whisker plot
- explicar os resultados dos plots
