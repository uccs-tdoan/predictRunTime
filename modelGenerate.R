
library(caret)
library(e1071)

setwd("C:/smallProject/predictRunTime")
df <- read.csv("TrainPCA.csv")
# excluding columns 1,3 and 24
df <- df[,-c(1,3,24)]
# Assume ther is a joint effect between Acc and Algorithm, we will create a dummy 
#variable for algorithm which generates binary attributes from nominal attributes
simpleMod <- dummyVars(~., data=df,levelsOnly= TRUE)

df <- predict(simpleMod, df)
# save to fiv
write.table(df,file="ModifiedTrainingPCA.csv",sep=",",row.names=FALSE,col.names=TRUE)    
# reload data 
df <- read.csv("TrainPCAPlot.csv")
# examine new data 
# plot data 
library(reshape2)
library(lattice)
# remove first, third  and 24th attribute
df <- df[,-c(1,3,24)]
meltedData <- melt(df,id.vars="Algorithm")
# plot density by algorithms factor, saved into file "densityplot" , folder graph
# plot indicates data mostly skew for each attributes
p<-densityplot(~value|variable,data = meltedData,scales = list(x = list(relation = "free"),y = list(relation = "free")),adjust = 1.25,pch = "|",xlab = "Predictor")

pdf("Densityplot.pdf", width=6, height=5)
print(p)
dev.off()
 
#check sknewness 
skew <- apply(df,2,skewness)
summary(skew)

#To create stratified random splits of the data (based on the classes)
# Set the random number seed so we can reproduce the results
df <- read.csv("ModifiedTrainingPCA.csv")
df <- df[,-c(1)]

set.seed(1)

# To apply linear regression, we transform data (Box-cox) for fix left, right skews
# trans <- preProcess(df, method = c("BoxCox"))
# transdf <- predict(trans, df)
# alternative, Yeo Johnson transformation to the data can be used
# tmp <- preProcess(df,method = "YeoJohnson")


# By default, the numbers are returned as a list. Using
# list = FALSE, a matrix of row numbers is generated.
# These samples are allocated to the training set.

TrainRow <- createDataPartition(df[, ncol(df)], p = 0.7, list= FALSE)

trainData <- df[TrainRow,]
ctrl <- trainControl(method = "repeatedcv", repeats = 5,number=10)

trainX <- df[TrainRow,1:ncol(df)-1]
trainY<-df[TrainRow,ncol(df)]
testX <- df[-TrainRow, 1:ncol(df)-1]
observed <- df[-TrainRow,ncol(df)]

# 1. Ordinary Linear Regression: RMSE = 2102.589 , after Box cox: 2.810313
 set.seed(215)
 lmFitAllPredictors <- lm(CPUtime ~ ., data = trainData)
 predlm <- predict(lmFitAllPredictors,testX)

 RMSE(predlm,observed)
 result <- cbind(observed,predlm)
 rm(predlm)
# problem: rank deficiency that leads to unreliable result 

 rm(trainData)


# 2 Use principle component regression. 
# RMSE= 5383.659 , after Box Cox transformed, RMSE =  4.638687
 set.seed(215)
 runPCR <- train(x = trainX, y = trainY, method = "pcr", trControl = ctrl,tuneLenght=25)
 predPCR <- predict(runPCR, newdata = testX)

 RMSE(predPCR,observed)
 result <- cbind(result,predPCR)
 rm(predPCR)

# 3 Use Partial Linear Regression. RMSE = 2746.797
 set.seed(215)
 runPLS <- train(x = trainX, y = trainY, method = "pls", preProcess=c("center","scale"),trControl = ctrl,tuneLenght=25)
 predPLS <- predict(runPLS, newdata = testX)

 RMSE(predPLS,observed)
 result <- cbind(result,predPLS)
 rm(predPLS)
# 4. Use Elastic net tuned over ve values of the L2 regularization parameter and the L1
#regularization parameter . RMSE=  2746.797
 set.seed(215)  
 library(elasticnet)
 enetGrid <- expand.grid(lambda = c(0, .001, .01, .1, 1), fraction = seq(.05, 1, length = 20))
 runENet <-  train(x = trainX, y = trainY, method = "enet", preProcess=c("center","scale"),trControl = ctrl,
 tuneGrid=enetGrid) 
 
 ## RMSE was used to select the optimal model using  the smallest value.
 ## The final values used for the model were fraction = 1 and lambda = 0. 
 enetModel <- enet(x = as.matrix(trainX), y = trainY, lambda = 0, normalize = TRUE)
 predEnet <- predict(enetModel, newx = as.matrix(testX), s = .1, mode = "fraction",    type = "fit")
  
 RMSE(predEnet$fit,observed)
 predENet <- predEnet$fit
 
 result <- cbind(result,predEnet)
 rm(predENet)  
# 5. RidgeRegression. RMSE = 2102.589 after Box Cox: RMSE 2.810313
 set.seed(215)
 ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
 ridgeRegFit <- train(trainX,trainY, method ="ridge",tuneGrid = ridgeGrid, trControl=ctrl, preProc=c("center","scale"))
 ridgeRegFit
#RMSE was used to select the optimal model using  the smallest value.
#The final value used for the model was lambda = 0.
 ridgeModel <- enet(x = as.matrix(trainX), y = trainY,lambda=0)
 ridgePred <- predict(ridgeModel,newx=as.matrix(testX),s=1,mode="fraction",type="fit")

 RMSE(ridgePred$fit,observed)
 predRidge <- ridgePred$fit 
 result <- cbind(result,predRidge)
 rm(predRidge)
 rm(ridgePred)  
 xyplot(trainY ~ predict(ridgeRegFit),type = c("p", "g"),xlab = "Predicted", ylab = "Observed")

# 6. SVM model, estimate cost value with tuneLength =10 in range 2^-2,...,2^-7, classification 
set.seed(215)  
library(kernlab)
svmRModel <- train(x = trainX,trainY,method = "svmRadial",preProc = c("center", "scale"), tuneLength = 10,trControl = trainControl(method = "cv")) 
#Tuning parameter 'sigma' was held constant at a value of 0.0108882
#RMSE was used to select the optimal model using  the smallest value.
#The final values used for the model were sigma = 0.0108882 and C = 16.
 svmRModel$finalModel
#SV type: eps-svr  (regression) 
#parameter : epsilon = 0.1  cost C = 8
 #svmFit <- ksvm(x = trainX, y = trainY,kernel ="rbfdot", kpar = "automatic",C = 8, epsilon = 0.1)

 
 predSVM <- predict(svmRModel, newdata = testX)
 RMSE(predSVM,observed)  # 1807.155 after Box Cox, RMSE : 1.393408 
 result <- cbind(result,predSVM)
 rm(predSVM)
# 7. MARS (Multivariate Adaptive Regression Splines)RMSE= 1739.119 after Box Cox 0.9171149
 set.seed(215)
 marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
 marsTuned <- train(trainX, trainY, method = "earth",tuneGrid = marsGrid,trControl = trainControl(method = "cv"))
 predMars <- predict(marsTuned,testX)
 RMSE(predMars,observed)
 result <- cbind(result,predMars)
 rm(predMars) 

# 7.1

marsTuned <- train(trainX, trainY, method = "bagEarth",tuneGrid = marsGrid,trControl = trainControl(method = "cv"))
# 8. K NN regression. RMSE = 2321.372 
# Remove a few sparse and unbalanced fingerprints first.
 set.seed(215)
 knnTuned <- train(trainX, trainY,method = "knn",preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:20),trControl = trainControl(method = "cv"))
 predKNN <- predict(knnTuned,testX)
 RMSE(predKNN, observed)
 result <- cbind(result,predKNN)
 rm(predKNN)
 # alternatively, postResample(pred = knnPred, obs = testY)
 write.table(df,file="preditedTimeModels.csv",sep=",",row.names=FALSE,col.names=TRUE)    
