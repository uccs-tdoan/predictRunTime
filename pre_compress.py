# Read all files to list for compression

from os import listdir
import numpy as np
import pandas as pd
from os.path import isfile, join
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, RandomizedPCA

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
  # Implement FA
   fa= FactorAnalysis(n_components=3)
   dat= fa.fit_transform(data)
   dat = pd.DataFrame(dat)
   
   dta = pd.concat([dat,tar],axis=1)
   dta.columns = header
   dta.to_csv(join('../FA','FA'+file))
  # Implement KPCA using np.column_stack((X, y)) to combine data and target
   kpca = KernelPCA(n_components=3,kernel="rbf", fit_inverse_transform=True, gamma=0.5)
   dat_kpca = kpca.fit_transform(data)
   dat_kpca = pd.DataFrame(dat_kpca)
   
   #d = np.column_stack((dat_kpca, target))
   dat_kpca = pd.concat([dat_kpca,tar],axis=1)
   
   dat = pd.DataFrame(dat_kpca)
   dat.columns = header
   
   dat.to_csv(join('../PCA','KPCA'+file))
      
 #Implement truncatedSVD
   tsvd = TruncatedSVD(3)
   tsvd.fit(data)
   dat = tsvd.transform(data)
   
   dat = pd.DataFrame(dat)
   #dat = np.column_stack((dat, target))
   dat= pd.concat([dat,tar],axis=1)
   
   dat.columns = header
   
   dat.to_csv(join('../PCA','TSVD'+file))

 #Implement RandomizedPCA
    
   rpca = RandomizedPCA(n_components=3)
   rpca.fit(data)   
   dat = rpca.transform(data)
   #X = scikit_pca.fit_transform(X)
   dat = pd.DataFrame(dat)
   
   #dat = np.column_stack((dat, target))
   dat = pd.DataFrame(dat)
   
   dat= pd.concat([dat,tar],axis=1)
   dat.columns = header
   
   dat.to_csv(join('../PCA','RPCA'+file))
 