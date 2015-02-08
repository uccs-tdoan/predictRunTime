# preprocess.py
# this current version processing one file each run. Enter file in csv format
# Enter column/ columns to convert from categories to binary
# Tri Doan
# Date: Feb 8, 2015 

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

def read_data(file_name,has_header=True):
    """
     read data 
    """
    f = open(file_name)
    header=[]
    if has_header: header=f.readline()
    data = []
    target = []
    for line in f:
        line = line.strip().split(",")
       # sample = [ float(x) if x number else x for x in line[:-1]]
        target.append(line[-1])
        data.append(line[:-1])
    return data,target,header

#for i in range(len(df.columns)-1):
#    if df.dtypes[i]==dtype('O') : print (' convert %d',i )

def nominal2Bin(data, column):
    dv = DictVectorizer()
    my_dict = [{'attr'+`column`: data[i][column]} for i in range(len(data))]
    return dv.fit_transform(my_dict).toarray()

print "Type the filename: "
file = raw_input("> ")
data,target,header  = read_data(join('..',file),True) 


print ('Enter a list of numbers, seperated by space ')
s = raw_input()
numbers = map(int, s.split())

df= np.column_stack((data, target))
step=0
for column in numbers:
   tmp = nominal2Bin(df, column+step)
   df = np.column_stack((df[:,:column],tmp,df[:,column+1:]))
   step= step+tmp.shape[1]+1


# replace new columns
df = pd.DataFrame(df)

