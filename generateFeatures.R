# R script for predict run time paper
# R script to compute statistical,theoretic summaries for datasets
library(foreign)
library("infotheo",lib="c:/r")
#library("moments",lib="c:/r")
library("psych",lib="c:/r")
library("PerformanceAnalytics")
# alternative, skewness function from moments library
# for PerformanceAnalytics, use skewness or SkewnessKurtosisRatio

#setwd("c:/smallProject/predictRunTime/code")
normClassEntropy = function(df) {
  targets = as.character(unique(df$Class))
  total_entropy=0
  for (i in targets)) {
       total_entropy = total_entropy + entropy(df[df$Class==i,5])
     }
  return (total_entropy/log2(ncol(df)))
}

# Compute the information in multiple features about the outcome
# Inputs: data frame, vector of feature numbers,
# number of target feature (optional, default=1)
# Calls: jt.entropy.columns
# Output: mutual information in bits
info.in.multi.columns = function(frame, feature.cols,
                                 target.col=1) {
  H.target = jt.entropy.columns(frame,target.col)
  H.features = jt.entropy.columns(frame,feature.cols)
  H.joint = jt.entropy.columns(frame,c(target.col,feature.cols))
  return(H.target + H.features - H.joint)
}

createMeta <- function (files) {
  n <- length(files)
  # general meta features
  instan <- numeric()        # number of instances in dataset
  classes <-  numeric()         # number of target classes 
  # statistical meta-features
     # Linear correlation coefficient is computed as corr(X,Y)
  LCoef01 <- numeric() # linearCorrCoef between attributes 0 and 1
  LCoef02 <- numeric() # linearCorrCoef between attributes 0 and 2
  LCoef03 <- numeric() # linearCorrCoef between attributes 0 and 3
  LCoef12 <- numeric() # linearCorrCoef between attributes 1 and 2
  LCoef13 <- numeric() # linearCorrCoef between attributes 1 and 3
  LCoef23 <- numeric() # linearCorrCoef between attributes 2 and 3
  
  FSize <- numeric()    # size of file in kb   
  skew0 <- numeric()
  skew1 <- numeric()
  skew2 <- numeric()
  skew3 <- numeric()
  kurtosis0 <- numeric()
  kurtosis1 <- numeric()
  kurtosis2 <- numeric()
  kurtosis3 <- numeric()
  
  # information-theoretic meta-features
  nCEntropy <- numeric()
  entroC <- numeric() # entropy between attributes and class
  toCorr <- numeric()  # known as mulitinformation
  # interinfo <- numeric  compute interinformation
  
  dataset <- character()
  for( i in  1:n) { 
        df <- read.csv(files[i])
        
        dataset[i] <- strsplit(files[i],'[.]')[[1]][1]
        instan[i] <- nrow(df)
        classes[i] <- length(unique(df$Class))
        FSize[i] <- object.size(df)/1000       
        LCoef01[i] <- cor(df$att0,df$att1)
        LCoef02[i] <- cor(df$att0,df$att2)
        LCoef03[i] <- cor(df$att0,df$att3)
        LCoef12[i] <- cor(df$att1,df$att2) 
        LCoef13[i] <- cor(df$att1,df$att3) 
        LCoef23[i] <- cor(df$att2,df$att3)
           
        skew0[i] <- skewness(df[,1])
        skew1[i] <- skewness(df[,2])
        skew2[i] <- skewness(df[,3])
        skew3[i] <- skewness(df[,4]) 
      
        kurtosis0[i] <- kurtosis(df[,1]) 
        kurtosis1[i] <- kurtosis(df[,2])
        kurtosis2[i] <- kurtosis(df[,3])
        kurtosis3[i] <- kurtosis(df[,4])
        
        # information-theoretics features
        
          # normalized class entropy indicates how much information needed to specify on class
        nCEntropy[i] <- entropy(df[ncol(df)],method="sg") /log2(ncol(df))
          # mutual information between attribute and class
        entroC[i] <- mutinformation(discretize(df[,-ncol(df)]),df[ncol(df)],method="sg") 
          # compute multiinformation (total correlation) for all attributes
        toCorr[i] <- multiinformation(discretize(df[,-ncol(df)]),method="sg")
    
    
  }
  dat<-data.frame(dataset,instan,classes,FSize,LCoef01, LCoef02, LCoef03, LCoef12,LCoef13, LCoef23,skew0,skew1,skew2,skew3, kurtosis0, kurtosis1, kurtosis2,kurtosis3,nCEntropy,entroC,toCorr)
  
  write.table(dat,file="../DataStats.csv",sep=",",row.names=FALSE,col.names=TRUE)
}

setwd("C:/smallProject/predictRunTime/compressDat") 
files <- list.files(getwd())
#n <- length(files)

createMeta(files)

