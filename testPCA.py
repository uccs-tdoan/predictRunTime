# Using PCA with specific dimension
# Ex: iris dataset in .data and .name format

#1. Linear PCA

#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import csv
from sklearn import decomposition
from sklearn import datasets
import numpy as np

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

#fig = plt.figure(1, figsize=(4, 3))
#plt.clf()
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#plt.cla()

pca = decomposition.PCA(n_components=3)

pca.fit(X)
X = pca.transform(X)
# simply use:  X = scikit_pca.fit_transform(X)

with open('c:\algoselecMeta\testPCA.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    a.write.rows(fp) 
fp.close()

# read csv file and return data, target    
# data should be numeric only
def read_csv(file_path, has_header = True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        target =[]
        for line in f:
            line = line.strip().split(",")
            data.append([float(x) for x in line[:-1]])
            target.append(line[-1])
    return data, target
    

def write_csv(file_path, data):
    with open(file_path,"w") as f:
        for line in data: f.write(",".join(line) + "\n")
        