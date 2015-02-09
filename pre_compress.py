# Pre_compress.py: implelement different reduction methods for dimensional reduction
#  including PCA, KernelPCA, FactorAnalysis, TruncatedSVD, RandomizedPCA, FastICA,
# MiniBatchSparsePCA, NMF
# Tri Doan
# Date: Jan 21, 2015 , last updated: Feb 9, 2015
 
import csv
import timeit
import numpy as np
import pandas as pd
from numpy.random import RandomState
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, RandomizedPCA, FastICA, MiniBatchSparsePCA, NMF
from sklearn.preprocessing import StandardScaler
# http://scikit-learn.org/stable/modules/decomposition.html

def read_csv(file_path, has_header = True):
    """
    read data from csv file path, file required to preprocess if needed and contains only numeric value
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

print ('Enter a number of dimension to reduce to: ')
s = raw_input()       
n_attr =int(s)

rng = RandomState(0)
csvfiles = [ f for f in listdir('../Data') if isfile(join('../Data',f)) and f.endswith(".csv")]

for file in csvfiles :
   runTime=[]
   data,target  = read_csv(join('../Data',file))    
   start = timeit.default_timer()
   
   # scale data 
   
   data_scaler = StandardScaler()
   target_scaler = StandardScaler()
   data = data_scaler.fit_transform(data)
   #target = target_scaler.fit_transform(target)
   
  # Implement Probabilistic PCA
   pca = PCA(n_components=n_attr) 
   pca.fit(data)
   dat = pca.transform(data)
      #data = np.concatenate([data,target],axis=1)
   # convert to Pandas data frame and write to file
   dat = pd.DataFrame(dat)
   tar = pd.DataFrame(target)
   dta = pd.concat([dat,tar],axis=1)
   header = ['att' +`i` for i in range(n_attr) ]+['Class']
   dta.columns = header
   dta.to_csv(join('../PCA','PCA'+`n_attr`+file),index=False)
   runTime.append(['PCA'+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
   
  # Implement Factor Analysis
   fa= FactorAnalysis(n_components=n_attr)
   dat= fa.fit_transform(data)
   dat = pd.DataFrame(dat)
   dta = pd.concat([dat,tar],axis=1)
   dta.columns = header
   dta.to_csv(join('../PCA','FA'+`n_attr`+file),index=False)
 
   runTime.append(['FA'+`n_attr`+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
     
    
  # Implement KernelPCA with rbf kernel
   start = timeit.default_timer()
   kpca = KernelPCA(n_components=n_attr,kernel="rbf", fit_inverse_transform=True, gamma=0.5)
   dat_kpca = kpca.fit_transform(data)
   dat_kpca = pd.DataFrame(dat_kpca)
   dat_kpca = pd.concat([dat_kpca,tar],axis=1)
  
   dat = pd.DataFrame(dat_kpca)
   dat.columns = header
   dat.to_csv(join('../PCA','KPCARBF'+`n_attr`+file),index=False)
   runTime.append(['KPCARBF'+`n_attr`+file,timeit.default_timer() - start ])
   start = timeit.default_timer()

# Implement KernelPCA with linear kernel
   start = timeit.default_timer()
   kpca = KernelPCA(n_components=n_attr,kernel="linear", fit_inverse_transform=True, gamma=0.5)
   dat_kpca = kpca.fit_transform(data)
   dat_kpca = pd.DataFrame(dat_kpca)
   dat_kpca = pd.concat([dat_kpca,tar],axis=1)
  

   dat = pd.DataFrame(dat_kpca)
   dat.columns = header
   dat.to_csv(join('../PCA','KPCAlinear'+`n_attr`+file),index=False)
   runTime.append(['KPCAlinear'+`n_attr`+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
  
 # Implement KernelPCA with sigmoid kernel
 #   start = timeit.default_timer()
 #  kpca = KernelPCA(n_components=n_attr,kernel="sigmoid", fit_inverse_transform=True, gamma=0.5)
 #  dat_kpca = kpca.fit_transform(data)
 #  dat_kpca = pd.DataFrame(dat_kpca)
 #  dat_kpca = pd.concat([dat_kpca,tar],axis=1)
  

 #  dat = pd.DataFrame(dat_kpca)
 #  dat.columns = header
 #  dat.to_csv(join('../PCA','KPCAsig'+`n_attr`+file),index=False)
 #  runTime.append(['KPCAsig'+`n_attr`+file,timeit.default_timer() - start ])
 #  start = timeit.default_timer()             

   
         
#Implement truncatedSVD
   start = timeit.default_timer()
   tsvd = TruncatedSVD(n_attr)
   tsvd.fit(data)
   dat = tsvd.transform(data)
   
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','TSVD'+`n_attr`+file),index=False)
   
   runTime.append(['TSVD'+`n_attr`+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
      
 #Implement RandomizedPCA
   start = timeit.default_timer() 
   rpca = RandomizedPCA(n_components=n_attr)
   rpca.fit(data)   
   dat = rpca.transform(data)
   #X = scikit_pca.fit_transform(X)
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','RPCA'+`n_attr`+file), index=False)

   runTime.append(['RPCA'+`n_attr`+file,timeit.default_timer() - start ])
   start = timeit.default_timer()
   
 # Implement FastICA
   start = timeit.default_timer() 
   fpca = FastICA(n_components=n_attr)
   fpca.fit(data)   
   dat = fpca.transform(data)
   dat = pd.DataFrame(dat)
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   dat.to_csv(join('../PCA','FICA'+`n_attr`+file),index=False)
   
   runTime.append(['FICA'+`n_attr`+file,timeit.default_timer() - start ])
 
   # implement SparsePCA
  
   # implement Nonnegetive Matrix Factorization NMF 
   #start = timeit.default_timer() 
   #snmfca = NMF(n_components=n_attr)
   #dat = snmfca.fit_transform(data)
   #dat = pd.DataFrame(dat)
   #dat= pd.concat([dat,tar],axis=1)
   #dat.columns = header
   #dat.to_csv(join('../PCA','NMFA'+`n_attr`+file))
   
   #runTime.append(['NMFA'+`n_attr`+file,timeit.default_timer() - start ])
   
  # csv_writer("../compresstime.csv",runTime)
    
   