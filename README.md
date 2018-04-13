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

#### Python script is on :file_folder: "irisdata.py"

## About Iris Dataset
 [[back to top](#project-instruction)]
 
  Iris dataset is a multivariate dataset of three classes of Irises and it was collected by the American botanist Edgar Anderson (1935) and introduced by the British statistician and geneticist Ronald Fisher in his article published in 1936 <i>"The Use of Multiple Measurements in Taxonomic Problems"<i> introducing linear-discriminant-function technique. The Iris data set is a best known and understood dataset and one of the most used to analyse data sets in statistics, data visualization, machine learning, etc. It is available in CSV format at Central Michigan University - CMU website (http://lib.stat.cmu.edu) [4, 6].The iris dataset is available online from University California Irvine's (UCI) machine-learning repository of datasets (http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) [1,2,3,5,6]. 
 "This is perhaps the best known database to be found in the pattern recognition literature. Fisher’s paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other."http://scikit-learn.org/stable/datasets/index.html#datasets


<img src="https://image.ibb.co/eJrdxH/Anderson_pres_1952.jpg" style="width: 700px; height:600px;"> <img src="https://image.ibb.co/jUYu4x/R_A_Fischer.jpg" style="width: 700px; height:600px;">

**Edgar Anderson and Ronald Fisher**

  
  The datasset contains data with 150 random samples of flowers of 50 samples from each of three species of irises (*setosa*, *versicolor* and *virginica*). Two parts of the flower were measured: sepal and pedal (lenght and width of each part in cm, total of four features)[1]. 
 
 <img src="https://image.ibb.co/jrWWjx/iris_petal_sepal.png">
**Iris Sepal and Petal measurement**
  
  Therefore, there are 5 attributes in the data base: sepal lenght in cm, sepal widht in cm, petal lenght in cm, petal width in cm and class - *iris setosa*, *iris versicolor* and *iris virginica* [2,4,6,7].
 The archive of Iris dataset can be view with any text editor and it contains 5 columns in each row: first four is the features and the fifth one is the label[1].
 
 <img src="https://image.ibb.co/kTbs2H/2018_04_11_13_05_03_sq_Es_Wbo_png_451_592.png">
 Iris Dataset characteristics. From: http://scikit-learn.org/stable/datasets/index.html#datasets

#### References
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

## My investigations (DRAFT)
[[back to top](#project-instruction)]

Before starting the project some programs, files and libraries must be downloaded and installed:
<ol>
 <li>Python version 3.6 downloaded via Anaconda3</li>
 <li>Visual Studio Code version 1.21.1 set up with Github<li> 
  <li>Iris dataset downloaded from UCI website and from other ways, such as via sklearn </li>
 <li>Libraries imported: scipy, numpy, matplotlib, pandas, sklearn, ESCREVER AQUI MAIS BIBLIOTECAS IMPORTADAS</li>
 </ol>
 
About some of the libraries imported: 
Pandas - for data-frame management package that allows for some useful function on the dataset.
Numpy - package useful for lineal algebra.
Matplotlib - good package to contruct visualizations.
Seaborn - great package to make the data visualizations very compelling.
Sklearn - package important to do machine learning in python.
http://www.philipkalinda.com/ds3.html


I've tried to convert the data from iris data set from strings to float numbers (def function make float) but it didn't work (the last columns couldn't be converted to float or number). 
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
 
 :clipboard: Note: the attempts to deal with Iris Dataset that didn't have sucess are not shown in the code block of 'irisdata.py' file.
 
  After so many attempts to get the min and max values of each column I've started to read more about the Iris Data Set and I've found an article about Iris Data Set and Machine Learning written by Jason Brownlee, Ph.D and he aimed to help professional developers to get started and confidently apply machine learning to address complex problems. In the article Jason Brownlee explains in a simple way about how to do machine learning using Python and he used Iris Data Set as an example using some key libraries , such as numpy, pandas, matplotlib, scipy and sklearn. In a easy and fast way I could analyse the data: getting descriptions (mean, min, max values and percentiles) of each attribute, creating histogram of each numerical variable. Moreover, I've found an online course about Pandas DataFrame on DataCamp and it help me to understand some code/functions and based on the videos I could create some of the codes, specially to get more info about the data set that I'm working on it. All these websites opened my mind to Pandas, Numpy, Matplot, etc. Then, it got easier to understand how I could get graphics in a few lines of code. Besides, reading some tutorials (Pyplot, Numpy, Pandas) I could improve my plots by giving them a title, axis label, etc.
  
  That way, after I read so many articles about Iris dataset and machine learning, Exploratory Data Analysis the following steps have been done:
  
:arrow_right: Print all styles available (Matplot)
  
  ```diff
  print(plt.style.available)
  ```
    
    Then, I ran simple descriptive staatistics on iris data set. That way, I could effectively appproach the iris data set.
      
  :arrow_right:Open the Iris dataset and each column got a 'header'
  
  ```diff 
  with open ('data/iris.csv') as f: #open iris data set
   names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
   dataset = pandas.read_csv(f, names=names)</b>
   ```
  
  :arrow_right:Information about the dataset: type, number of rows and columns, name of the columns, pandas index, datatimeindex(number of columns, type of data in each column, ...). Moreover, to get a general idea about the dataset and to take a closer look at the data itself the function  (dataset.tail(20)) was used to check out the last 20 rows. That way, I got a general idea about the data set.
  
   ```diff 
  print(type(dataset)) 
  print(dataset.shape) 
  print(dataset.columns)
  print(type(dataset.columns))
  print(dataset.index) 
  print(dataset.tail(20)) 
  print(dataset.info()) 
  ```
  :arrow_right:Summary statistics that exclude NaN values -  table that summarises the numeric information in the dataset, such as count, mean, standard deviation, minimum and maximum values and the quantiles of the data. It  was useful to gain a general sense of how the data is structured and about the data per se. 
  
  ```diff 
  print(dataset.describe())
   ```
  
  Pivot table with the variables means ---FALAR SOBRE O QUE EH PIVOT TABEL
 print(dataset.pivot_table(index='class', values=[
      'sepal-length', 'sepal-width', 'petal-length', 'petal-width'], aggfunc=np.mean))

  
  :arrow_right:Plots
  
  :arrow_right_hook: Histogram: 'A histogram is a plot that lets you discover, and show, the underlying frequency distribution (shape) of a set of continuous data. This allows the inspection of the data for its underlying distribution (e.g., normal distribution), outliers, skewness, etc'[7].
  
  <img scr="https://image.ibb.co/hy1BUx/iris_hist_sepallenght.png"><img src="https://image.ibb.co/cq9y9x/iris_hist_sepalwidth.png">
  <img src="https://image.ibb.co/kjvuhH/iris_hist_petallenght.png"><img src="https://image.ibb.co/jPrBUx/iris_hist_petalwidth.png">
  
  :arrow_right_hook:Plot on Dataframe: 'is a convenience to plot all of the columns with labels'
  
  :arrow_right_hook: Multiple bar plot
  <img src="https://image.ibb.co/j3O0cH/Figure_1.png">
  
 :arrow_right_hook:Scatter plot: A scatter plot is a graph in which a set of points plotted on a horizontal and vertical axes.As scatter plots show the extent of correlation between the values of the variables, they are an important tool in statistics. If there is no correlation between the variables the points will show randomly scattered on the coordinate plane. But, if there is a large correlation between the variables, the points concentrate near a straight line. Therefore, scatter plots are quite useful for data visualization as they illustrate a trend. Scatter plots shows not only the extent of correlation, but also the sense of the correlation. Neverthless, scatter plots not show the causation. 
  So, if the vertical (called y axis) variable increases and the horizontal (called x axis) variable also increases it means that there is a correlation (positive correlarion). The maximum positive correlation that is possible is +100% or +1 (all points in the plot lie along a straight line in a positive slope). In case the y axis variable decreases and the x axis increases or vice-versa it is a negative correlation. The maximum negative correlation that is possible is -100% or -1 (all the points in the plot lie along a straight line in a negative slope).  
-------->https://whatis.techtarget.com/definition/scatter-plot
  
  **Linear transformation techniques** - Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA)
  
:arrow_forward: PCA: identifies the combination of attributes (principal components, or directions in the feature space) that account for the most variance in the data. It is a linear dimensionality reduction that uses Singular Value Decomposition of the data of dataset to a lower dimensional space. In this case, samples of the two fisrt components were plotted. Usually, PCA is used to speed up a machine learning algorithm, helping to visualize your data. As Iris dataset is 4 dimensional the data must be projects into 2, that way it is easy to understand better the data and it is easier to plot. After the reduction from 4 to 2 dimension data usually there is no particular meaning designated to each main component. 
 
  <img src="https://image.ibb.co/kRLD9x/PCAIris.png">
  
 :arrow_forward: LDA: indentifies attributes that account for the most variance between classes. LDA In particular, LDA, in contrast to PCA, is a supervised method, using known class labels.
 <img src="https://image.ibb.co/fqX4Gc/LDAIris.png">
 
 :heavy_exclamation_mark:PCA vs LDA: Both LDA and PCA are used for dimensionality reduction, but PCA is described as an unsupervised method, because it does not take into account class labels and its object is to find the directions, known as principal components that maximize the variance in a dataset. While LDA is supervised method and computes the directions (“linear discriminants”) that will represent the axes that that maximize the separation between multiple classes.
  
[[back to top](#project-instruction)]

-----no comeco do projeto escrever issoo: This dataset can be approached in a number of different ways and this will depend on your objectives when starting out your analysis. I shall highlight the objectives later in the analysis. Let’s just describe the data so we know what we’re dealing with.
http://www.philipkalinda.com/ds3.html

------After graphing the features in a pair plot, it is clear that the relationship between pairs of features of a iris-setosa (in pink) is distinctly different from those of the other two species.
There is some overlap in the pairwise relationships of the other two species, iris-versicolor (brown) and iris-virginica (green).
https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset

LEMBRETE PARA MIM MESMA
- ajeitar as images
- ajeitar as legendas
- atualizar e ajeitar as referencias....fazer nova referencia para textos depois da introducao?
- rever o inicio da minha investigacao com as coisas necessarias e descrever mais cada um 
- rever todas as bibliotecas que importei no codigo. usei todas?
- terminar de explicar sobre cada tipo de analise feita, suas definicoes
- explicar os resultados dos plots

