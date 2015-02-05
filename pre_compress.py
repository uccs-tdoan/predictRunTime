# Read all files to list for compression

from os import listdir
import numpy as np
import pandas as pd
from os.path import isfile, join
from sklearn import decomposition

def read_csv(file_path, has_header = True):

    with open(file_path) as f:
        if has_header:  f.readline()
        data = []
        target =[]
        for line in f:
            line = line.strip().split(",")
            data.append([float(x) for x in line[:-1]])
            target.append([line[-1]])
    return data, target

n_attr =3
csvfiles = [ f for f in listdir('../Data') if isfile(join('../Data',f)) and f.endswith(".csv")]

for file in csvfiles :
   data,target  = read_csv(join('../Data',file))    
   pca = decomposition.PCA(n_components=n_attr) 
   pca.fit(data)
   
   
   data = pca.transform(data)
    
   #data = np.concatenate([data,target],axis=1)
   
   # convert to Pandas data frame and write to file
   data = pd.DataFrame(data)
   target = pd.DataFrame(target)
   data = pd.concat([data,target],axis=1)
   header = ['att' +`i` for i in range(n_attr) ]+['Class']
   data.columns = header
   data.to_csv(join('../PCA',file))
   
   #X = scikit_pca.fit_transform(X)
    