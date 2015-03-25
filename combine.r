---
title: "combineR"
author: "tri doan"
date: "Wednesday, March 25, 2015"
output: pdf_document
---
 1. merge two data data frames (Performance3,Performance4) whichbased on dataset
 which contains only CPUtime, algorithm, ... from a cluster of Virtual Machine  implement different mining algorithms from API weka
 2. Merge with DataStats.csv which contains statistic information to generate final data for regression model
 Current folder is predictRunTime/code   
 Note: Code is in code folder; datas are places in rawexperiments folder
 output: file is saved in predictRuntime
 
```{r, echo=FALSE}
 library(plyr) 
 csv.files <- dir(path = "../rawexperiments", pattern = "csv$", full.names=TRUE) 
 data <- ldply(csv.files, read.csv)

```
Then we merge this file with DataStats.csv to generate full data (some attributes unused in this paper such as usedMemory, SAR) that can be removed. 

```{r, echo=FALSE}
 df <- read.csv("../DataStats.csv")
 total <- merge(df,data, by="dataset")
 write.csv(total, file = "../TrainPCA.csv",row.names=FALSE)

plot(cars)
```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
