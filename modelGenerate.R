
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
# To apply linear regression, we transform data (Box-cox) for fix left, right skews
# alternative, Yeo Johnson transformation to the data can be used
#tmp <- preProcess(df,method = "YeoJohnson")

#To create stratified random splits of the data (based on the classes)
# Set the random number seed so we can reproduce the results
df <- read.csv("ModifiedTrainingPCA.csv")

set.seed(1)

# By default, the numbers are returned as a list. Using
# list = FALSE, a matrix of row numbers is generated.
# These samples are allocated to the training set.

Training <- createDataPartition(df[, ncol(df)], p = 0.7, list= FALSE)

trainX <- df[Training,1:ncol(df)-1]
trainY<-df[Training,ncol(df)]
testX <- df[-Training, 1:ncol(df)-1]
testY <- df[-Training,ncol(df)]

trainData <- df[Training,]
ctrl <- trainControl(method = "repeatedcv", repeats = 5,number=10)

# 1. Ordinary Linear Regression: 
set.seed(215)
lmFitAllPredictors <- lm(CPUtime ~ ., data = trainData)
lmPredict <- predict(lmFitAllPredictors,testX)
RMSE(lmPredict,testY)

# problem: rank deficiency that leads to unreliable result

# 2 Use principle component regression 
set.seed(215)
runPCR <- train(x = trainX, y = trainY, method = "pcr", trControl = ctrl,tuneLenght=25)
predicted <- predict(runPCR, newdata = testX)
observed <- testY
RMSE(predicted,observed)

# 3 Use Partial Linear Regression
set.seed(215)
runPLS <- train(x = trainX, y = trainY, method = "pls", preProcess=c("center","scale"),trControl = ctrl,tuneLenght=25)
predicted <- predict(runPLS, newdata = testX)
observed <- testY
RMSE(predicted,observed)


# 4. Use Elastic net tuned over ve values of the L2 regularization parameter and the L1
#regularization parameter
 set.seed(215)  
 library(elasticnet)
 enetGrid <- expand.grid(lambda = c(0, .001, .01, .1, 1), fraction = seq(.05, 1, length = 20))
 runENet <-  train(x = trainX, y = trainY, method = "enet", preProcess=c("center","scale"),trControl = ctrl,
 tuneGrid=enetGrid) 

 ## RMSE was used to select the optimal model using  the smallest value.
 ## The final values used for the model were fraction = 1 and lambda = 0. 
 enetModel <- enet(x = as.matrix(trainX), y = trainY, lambda = 0, normalize = TRUE)
 enetPred <- predict(enetModel, newx = as.matrix(testX), s = .1, mode = "fraction",    type = "fit")
 observed <- testY
 RMSE(predicted,observed)


# 5. RidgeRegression 
 set.seed(215)
 ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
 ridgeRegFit <- train(trainX,trainY, method ="ridge",tuneGrid = ridgeGrid, trControl=ctrl, preProc=c("center","scale"))
 ridgeRegFit
#RMSE was used to select the optimal model using  the smallest value.
#The final value used for the model was lambda = 0.
 ridgeModel <- enet(x = as.matrix(trainX), y = trainY,lambda=0)
 ridgePred <- predict(ridgeModel,newx=as.matrix(testX),s=1,mode="fraction",type="fit")
 observed <- testY
 RMSE(ridgePred$fit,observed)

 xyplot(trainY ~ predict(ridgeRegFit),type = c("p", "g"),xlab = "Predicted", ylab = "Observed")

# 6. SVM model, estimate cost value with tuneLength =10 in range 2^-2,...,2^-7, classification 
set.seed(215)  
library(kernlab)
 svmRTuned <- train(trainX, trainY, method = "svmRadial",preProc = c("center", "scale"),tuneLength=10, trControl = trainControl(method = "cv"))
#Tuning parameter 'sigma' was held constant at a value of 0.0108882
#RMSE was used to select the optimal model using  the smallest value.
#The final values used for the model were sigma = 0.0108882 and C = 16.
 svmRTuned$finalModel
#SV type: eps-svr  (regression) 
#parameter : epsilon = 0.1  cost C = 16 
 svmFit <- ksvm(x = trainX, y = trainY,kernel ="rbfdot", kpar = "automatic",C = 16, epsilon = 0.1)

 svmRModel <- train(x = trainX,trainY,method = "svmRadial",preProc = c("center", "scale"), tuneLength = 8)
 svmRPred <- predict(svmRModel, newdata = testX)
 RMSE(svmRPred,observed)
# 7. MARS , Multivariate Adaptive Regression Splines
 set.seed(215)
 marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
 marsTuned <- train(trainX, trainY, method = "earth",tuneGrid = marsGrid,trControl = trainControl(method = "cv"))
 marsPred <- predict(marsTuned,testX)
 RMSE(marsPred,observed)
 

# 9. K NN regression 
# Remove a few sparse and unbalanced fingerprints first
 knnDescr <- trainX[, -nearZeroVar(trainX)]
 knnTune <- train(trainX, trainY,method = "knn",preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:20),trControl = trainControl(method = "cv"))
 knnPred <- predict(knnTune,testX)
 RMSE(knnPred, observed)
 # alternatively, postResample(pred = knnPred, obs = testY)