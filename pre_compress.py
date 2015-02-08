# Read all files to list for compression

import csv
import timeit
import numpy as np
import pandas as pd
from numpy.random import RandomState
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, RandomizedPCA, FastICA, MiniBatchSparsePCA, NMF
# http://scikit-learn.org/stable/modules/decomposition.html

def read_csv(file_path, has_header = True):
    """
    read data from csv file path
    """
    with open(file_path) as f:
        if has_header:  f.readline()
        data = []
        target =[]
        for line in f:
            line = line.strip().split(",")
            data.append([float(x) for x in line[:-1]])
            target.append([line[-1]])
    return data, target

def csv_writer(file_path, data):
    """
    Write data to a CSV file path
    """
    with open(file_path, "a+") as f:
        #writer = csv.writer(f, delimiter=',')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
    f.close()        
    
n_attr =3
rng = RandomState(0)
csvfiles = [ f for f in listdir('../Data') if isfile(join('../Data',f)) and f.endswith(".csv")]

for file in csvfiles :
   runTime=[]
   data,target  = read_csv(join('../Data',file))    
   start = timeit.default_timer()
  # Implement PCA
   pca = PCA(n_components=n_attr) 
   kernel = KernelPCA(n_components=n_attr) 
   pca.fit(data)
   dat = pca.transform(data)
      #data = np.concatenate([data,target],axis=1)
   # convert to Pandas data frame and write to file
   dat = pd.DataFrame(dat)
   tar = pd.DataFrame(target)
   dta = pd.concat([dat,tar],axis=1)
   header = ['att' +`i` for i in range(n_attr) ]+['Class']
   dta.columns = header
   dta.to_csv(join('../PCA','PCA'+file))
   runTime.append(['PCA'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
   
  # Implement Factor Analysis
   fa= FactorAnalysis(n_components=3)
   dat= fa.fit_transform(data)
   dat = pd.DataFrame(dat)
   dta = pd.concat([dat,tar],axis=1)
   dta.columns = header
   dta.to_csv(join('../PCA','FA'+file))
 
   runTime.append(['FA'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
     
    
  # Implement KernelPCA 
   start = timeit.default_timer()
   kpca = KernelPCA(n_components=3,kernel="rbf", fit_inverse_transform=True, gamma=0.5)
   dat_kpca = kpca.fit_transform(data)
   dat_kpca = pd.DataFrame(dat_kpca)
   dat_kpca = pd.concat([dat_kpca,tar],axis=1)
  
   dat = pd.DataFrame(dat_kpca)
   dat.columns = header
   dat.to_csv(join('../PCA','KPCA'+file))
   runTime.append(['KPCA'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()

         
 #Implement truncatedSVD
   start = timeit.default_timer()
   tsvd = TruncatedSVD(3)
   tsvd.fit(data)
   dat = tsvd.transform(data)
   
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','TSVD'+file))
   
   runTime.append(['TSVD'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
      
 #Implement RandomizedPCA
   start = timeit.default_timer() 
   rpca = RandomizedPCA(n_components=3)
   rpca.fit(data)   
   dat = rpca.transform(data)
   #X = scikit_pca.fit_transform(X)
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','RPCA'+file))

   runTime.append(['RPCA'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
   
 # Implement FastICA
   start = timeit.default_timer() 
   fpca = FastICA(n_components=3)
   fpca.fit(data)   
   dat = fpca.transform(data)
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','FICA'+file))
   
   runTime.append(['FICA'+file,timeit.default_timer() - start ])
 
   # implement SparsePCA
  
   # implement Nonnegetive Matrix Factorization NMF 
   start = timeit.default_timer() 
   snmfca = NMF(n_components=3)
   dat = snmfca.fit_transform(data)
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','NMFA'+file))
   
   runTime.append(['NMFA'+file,timeit.default_timer() - start ])
   
   csv_writer("../compresstime.csv",runTime)
    
   