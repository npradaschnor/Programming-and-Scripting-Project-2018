# Programming-and-Scripting-Project-2018
###### Project instruction
The project entails you researching the data set, and then writing documentation and code in the Python programming language based on that research. An online search for information on the data set will convince you that many people have investigated and written about it previously, and many of those are not experienced programmers. You are expected to be able to break this project into several smaller tasks that are easier to solve, and to plug these together after they have been completed. 
###### You might do that for this project as follows:
- [x] Research background information about the data set and write a summary about it
- [ ] Keep a list of references you used in completing the project
- [ ] Download the data set and write some Python code to investigate it
- [ ] Summarise the data set by, for example, calculating the maximum, minimum and mean of each column of the data set. A Python script will quickly do this for you
- [ ]  Write a summary of your investigations
- [ ] Include supporting tables and graphics as you deem necessary

## About Iris Dataset
 
 
  Iris dataset is a multivariate dataset of three classes of Irises and it was collected by the American botanist Edgar Anderson (1935) and introduced by the British statistician and geneticist Ronald Fisher in his article published in 1936 <i>"The Use of Multiple Measurements in Taxonomic Problems"<i> introducing linear-discriminant-function technique. The Iris data set is a best known and understood dataset and one of the most used to analyse data sets in statistics, data visualization, machine learning, etc. It is available in CSV format at Central Michigan University - CMU website (http://lib.stat.cmu.edu) [4, 6].The iris dataset is available online from University California Irvine's (UCI) machine-learning repository of datasets (http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) [1,2,3,5,6]. 

**Edgar Anderson (IMG)

**Ronald Fisher (IMG)
  
  The datasset contains data with 150 random samples of flowers of 50 samples from each of three species of irises (*setosa*, *versicolor* and *virginica*). Two parts of the flower were measured: sepal and pedal (lenght and width of each part in cm, total of four features)[1]. 
 
 **Iris Sepal and Petal measurement (IMG)
  
  Therefore, there are 5 attributes in the data base: sepal lenght in cm, sepal widht in cm, petal lenght in cm, petal width in cm and class - *iris setosa*, *iris versicolor* and *iris virginica* [2,4,6,7].
 The archive of Iris dataset can be view with any text editor and it contains 5 columns in each row: first four is the features and the fifth one is the label[1].

#### References
<ol type="[1]">
  <li> Predictive Analytics For Dummies Available: http://www.dummies.com/DummiesTitle/Predictive-Analytics-For-Dummies.productCd-1118728963,navId-322439,descCd-DOWNLOAD.html</li>
  <li>https://www.lynda.com/Apache-Spark-tutorials/Preprocessing-Iris-data-set/559180/674634-4.html</li>
  <li>http://www.fon.hum.uva.nl/praat/manual/iris_data_set.html</li>
  <li>http://www.idvbook.com/teaching-aid/data-sets/the-iris-data-set/</li>
  <li>http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf</li>
  <li>https://technichesblog.wordpress.com/2015/10/25/matlab-code-to-import-iris-data/</li>
  <li>https://statistics.laerd.com/statistical-guides/understanding-histograms.php</li>
  </ol>

[[back to top](#project-instruction)]

## My investigations (DRAFT)

Before starting the project some programs, files and libraries must be downloaded and installed:
<ol>
 <li>Python v 3.6</li>
 <li>Libraries: scipy, numpy, matplotlib, pandas, sklearn</li>
 <li>download iris dataset from UCI website</li>
 </ol>
 
First I've tried to convert the data from iris data set from strings to float numbers (def function make float) but it didn't work (the last columns couldn't be converted to float or number). 
```diff 
def makefloat(teststr):
 try:
  return float(teststr)  # return a floating point value if possibe.
 except:
  return teststr  # otherwise return the string as is*
 ```   
Every time I tried to deal with dataset I couldn't because the last column contains text. Then, I've tried to delete the last column (named class), so I could get a dataset consisted only with numbers using pandas as it follows.

```diff 
data = pandas.read_csv('data/iris.csv')  # select csv file
data = data.drop(['class'], axis=1)  # delete the column named 'class'
data.to_csv('data/datairis.csv') # create new file (that not contains the column 'class')*
```
After I got a dataset that contained only numbers I tried to find a way to find the min and max values of each column. But every code that I've tried didn't work.

```diff 
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
  
  After so many attempts to get the min and max values of each column I've started to read more about the Iris Data Set and I've found an article about Iris Data Set and Machine Learning written by Jason Brownlee, Ph.D and he aimed to help professional developers to get started and confidently apply machine learning to address complex problems. In the article Jason Brownlee explains in a simple way about how to do machine learning using Python and he used Iris Data Set as an example using some key libraries , such as numpy, pandas, matplotlib, scipy and sklearn. In a easy and fast way I could analyse the data: getting descriptions (mean, min, max values and percentiles) of each attribute, creating histogram of each numerical variable. Moreover, I've found an online course about Pandas DataFrame on DataCamp and it help me to understand some code/functions and based on the videos I could create some of the codes, specially to get more info about the data set that I'm working on it. All these websites opened my mind to Pandas, Numpy, Matplot, etc. Then, it got easier to understand how I could get graphics in a few lines of code. Besides, reading some tutorials (Pyplot, Numpy, Pandas) I could improve my plots by giving them a title, axis label, etc.
  
  That way, after I read so many articles about Iris dataset and machine learning, Exploratory Data Analysis the following steps have been done:
  --> Print all styles available (Matplot)
  
  ```diff
  print(plt.style.available)
  ```
    
  --> Open the Iris dataset and each column got a 'header'
  
  ```diff 
  with open ('data/iris.csv') as f: #open iris data set
   names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
   dataset = pandas.read_csv(f, names=names)</b>
   ```
  
  --> Information about the dataset: type, number of rows and columns, name of the columns, pandas index, datatimeindex(number of columns, type of data in each column, ...). Moreover, to get a general idea about the dataset and to take a closer look at the data itself the function  (dataset.tail(20)) was used to check out the last 20 rows.
  got a general idea about your data set
  
  ```diff 
  print(type(dataset)) 
  print(dataset.shape) 
  print(dataset.columns)
  print(type(dataset.columns))
  print(dataset.index) 
  print(dataset.tail(20)) 
  print(dataset.info()) 
  ```
  --> Summary statistics that exclude NaN values. Return the count, mean, standard deviation, minimum and maximum values and the quantiles of the data.
  
  ```diff 
  print(dataset.describe())
   ```
  --> Plots
  
  - Histogram: 'A histogram is a plot that lets you discover, and show, the underlying frequency distribution (shape) of a set of continuous data. This allows the inspection of the data for its underlying distribution (e.g., normal distribution), outliers, skewness, etc'[7].
  
  - Plot on Dataframe: 'is a convenience to plot all of the columns with labels'
  
  - Multiple bar plot
  
  
  
