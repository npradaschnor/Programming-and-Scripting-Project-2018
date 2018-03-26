#Noa P Prada Schnor 2018-03-24

import pandas
# Pandas - library highlight: 1."Tools for reading and writing data between in-memory data structures and different formats: CSV and text files, Microsoft Excel, SQL databases, and the fast HDF5 format", 2."Columns can be inserted and deleted from data structures for size mutability;" https://pandas.pydata.org/; - As I want to delete one column from my csv file I decided to import /use pandas.

#I named all the columns and then deleted the last column named 'class'. I've done that because the columns contined text(string) not numbers. That way, I can deal easier with numbers. The original file with data iris set is data/iris.csv and the new file created (without the last column) is data/datairis.csv

data = pandas.read_csv('data/iris.csv') #select the file iris.csv
data=data.drop(['class'],axis=1) #delete the column named 'class'
data.to_csv('data/datairis.csv') #create new file that will not contain the column 'class'
