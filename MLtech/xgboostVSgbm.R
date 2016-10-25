#COMPARE XGBOOST with GBM
### Packages Required

library(caret)
library(corrplot)			# plot correlations
library(doParallel)		# parallel processing
library(dplyr)        # Used by caret
library(gbm)				  # GBM Models
library(pROC)				  # plot the ROC curve
library(xgboost)      # Extreme Gradient Boosting


### Get the Data
# Load the data and construct indices to divied it into training and test data sets.

data(segmentationData)  	# Load the segmentation data set
dim(segmentationData)
head(segmentationData,2)
#
trainIndex <- createDataPartition(segmentationData$Case,p=.5,list=FALSE)
trainData <- segmentationData[trainIndex,-c(1,2)]
testData  <- segmentationData[-trainIndex,-c(1,2)]
#
trainX <-trainData[,-1]        # Pull out the dependent variable
testX <- testData[,-1]
sapply(trainX,summary) # Look at a summary of the training data

## GENERALIZED BOOSTED RGRESSION MODEL (BGM)  

# Set up training control
ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation
                     number = 5,							# do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     allowParallel = TRUE)

# Use the expand.grid to specify the search space	
# Note that the default search grid selects multiple values of each tuning parameter

grid <- expand.grid(interaction.depth=c(1,2), # Depth of variable interactions
                    n.trees=c(10,20),	        # Num trees to fit
                    shrinkage=c(0.01,0.1),		# Try 2 values for learning rate 
                    n.minobsinnode = 20)
#											
set.seed(1951)  # set the seed

# Set up to do parallel processing   
registerDoParallel(4)		# Registrer a parallel backend for train
getDoParWorkers()

gbm.tune <- train(x=trainX,y=trainData$Class,
                  method = "gbm",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid=grid,
                  verbose=FALSE)


# Look at the tuning results
# Note that ROC was the performance criterion used to select the optimal model.   

gbm.tune$bestTune
plot(gbm.tune)  		# Plot the performance of the training models
res <- gbm.tune$results
res

### GBM Model Predictions and Performance
# Make predictions using the test data set
gbm.pred <- predict(gbm.tune,testX)

#Look at the confusion matrix  
confusionMatrix(gbm.pred,testData$Class)   

#Draw the ROC curve 
gbm.probs <- predict(gbm.tune,testX,type="prob")
head(gbm.probs)

gbm.ROC <- roc(predictor=gbm.probs$PS,
               response=testData$Class,
               levels=rev(levels(testData$Class)))
gbm.ROC$auc
#Area under the curve: 0.8731
plot(gbm.ROC,main="GBM ROC")

# Plot the propability of poor segmentation
histogram(~gbm.probs$PS|testData$Class,xlab="Probability of Poor Segmentation")
##----------------------------------------------
## XGBOOST
# Some stackexchange guidance for xgboost
# http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees

# Set up for parallel procerssing
set.seed(1951)
registerDoParallel(4,cores=4)
getDoParWorkers()

# Train xgboost
xgb.grid <- expand.grid(nrounds = 500, #the maximum number of iterations
                        eta = c(0.01,0.1), # shrinkage
                        max_depth = c(2,6,10))

xgb.tune <-train(x=trainX,y=trainData$Class,
                 method="xgbTree",
                 metric="ROC",
                 trControl=ctrl,
                 tuneGrid=xgb.grid)


xgb.tune$bestTune
plot(xgb.tune)  		# Plot the performance of the training models
res <- xgb.tune$results
res

### xgboostModel Predictions and Performance
# Make predictions using the test data set
xgb.pred <- predict(xgb.tune,testX)

#Look at the confusion matrix  
confusionMatrix(xgb.pred,testData$Class)   

#Draw the ROC curve 
xgb.probs <- predict(xgb.tune,testX,type="prob")
#head(xgb.probs)

xgb.ROC <- roc(predictor=xgb.probs$PS,
               response=testData$Class,
               levels=rev(levels(testData$Class)))
xgb.ROC$auc
# Area under the curve: 0.8857

plot(xgb.ROC,main="xgboost ROC")
# Plot the propability of poor segmentation
histogram(~xgb.probs$PS|testData$Class,xlab="Probability of Poor Segmentation")


# Comparing Multiple Models
# Having set the same seed before running gbm.tune and xgb.tune
# we have generated paired samples and are in a position to compare models 
# using a resampling technique.
# (See Hothorn at al, "The design and analysis of benchmark experiments
# -Journal of Computational and Graphical Statistics (2005) vol 14 (3) 
# pp 675-699) 

rValues <- resamples(list(xgb=xgb.tune,gbm=gbm.tune))
rValues$values
summary(rValues)

bwplot(rValues,metric="ROC",main="GBM vs xgboost")	# boxplot
dotplot(rValues,metric="ROC",main="GBM vs xgboost")	# dotplot
#splom(rValues,metric="ROC")