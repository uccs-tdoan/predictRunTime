# ImputeMissing.py
# preprocessing missing numeric values by mean 
# Data contains numeric value ony
# Tri Doan
# Date: Feb 9, 2015
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
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
         # convert numeric only value
        sample = [ float(x) if x.replace('.','',1).isdigit() else np.nan for x in line[:-1]]
        target.append(line[-1])
        data.append(sample)
    return data,target,header


print "Type the filename: "
file = raw_input("> ")

data,target,header  = read_data(join('..',file),True)

mask = np.ma.masked_array(data, np.isnan(data))
data[mask_t] = np.nan

# alternative, use strategy='median'
impute = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_new = impute.fit_transform(data)

df= np.column_stack((data_new, target))
# covert to DataFrame
df = pd.DataFrame(df)
df.columns= header.strip().split(",")
# save to file without index column
df.to_csv(join('../tmp',file),index=False)
