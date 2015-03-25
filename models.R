---
title: "Untitled"
output: pdf_document
---

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

```{r,echo=FALSE}
library(caret)
library(e1071)
library(ggplot2)
setwd("C:/smallProject/predictRunTime")

```

In this code section, we create ModifiedTrainingPCA.csv from TrainPCA.csv where category 
variable (Algorithm) will be converted into binary atribute (similar nominal to binary in weka)


```{r, echo=FALSE}
df <- read.csv("TrainPCA.csv")
simpleMod <- dummyVars(~., data=df,levelsOnly= TRUE)

df <- predict(simpleMod, df)
write.table(df,file="ModifiedTrainingPCA.csv",sep=",",row.names=FALSE,col.names=TRUE)    
```
 In this code section, we plot density by algorithm of TrainPCAplot.csv which
indicates skewly data for most attributes

```{r, echo=FALSE}
df <- read.csv("TrainPCAPlot.csv")
library(reshape2)
library(lattice)

df <- df[,-c(1,3,24)]
meltedData <- melt(df,id.vars="Algorithm")
p<-densityplot(~value|variable,data = meltedData,scales = list(x = list(relation = "free"),y = list(relation = "free")),adjust = 1.25,pch = "|",xlab = "Predictor")

pdf("Densityplot.pdf", width=6, height=5)
print(p)
dev.off()

```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
