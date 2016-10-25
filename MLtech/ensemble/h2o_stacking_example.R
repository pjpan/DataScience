# Train 4 models and ensemble them together with new h2o.stack function.

# Requirements: Models must be same type of model H2OBinomial, etc
# Must have same outcome
# Must have used `fold_assignment = "Modulo"` and same number for `nfolds`, 
# or identical `fold_column` must be used to guarantee same folds between base models

# Requires: cvpreds.R and stack.R
source("https://gist.githubusercontent.com/ledell/f3a87bd136ce06e0a5ff/raw/2a82535892ff66694a1a401de46b8b5a92820849/cvpreds.R")
source("https://gist.githubusercontent.com/ledell/f389ac1e9c6e7000b299/raw/6bc1d2c9cfe1a51ffcdcf79cf184e80a40d4828f/stack.R")


library(h2oEnsemble)  # Requires version >=0.0.4 of h2oEnsemble
library(cvAUC)  # Used to calculate test set AUC (requires version >=1.0.1 of cvAUC)
localH2O <-  h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine

# Import a sample binary outcome train/test set into H2O
h2o.init(nthreads = -1)
train_csv <- "https://h2o-public-test-data.s3.amazonaws.com/smalldata/testng/higgs_train_5k.csv"
test_csv <- "https://h2o-public-test-data.s3.amazonaws.com/smalldata/testng/higgs_test_5k.csv"
train <- h2o.importFile(train_csv)
test <- h2o.importFile(test_csv)
y <- "response"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
family <- "binomial"
nfolds <- 5


glm1 <- h2o.glm(x = x, y = y, family = family, 
                training_frame = train,
                nfolds = nfolds,
                fold_assignment = "Modulo",
                keep_cross_validation_predictions = TRUE)

gbm1 <- h2o.gbm(x = x, y = y, distribution = "bernoulli",
                training_frame = train,
                nfolds = 5,
                fold_assignment = "Modulo",
                keep_cross_validation_predictions = TRUE)

dl1 <- h2o.deeplearning(x = x, y = y, distribution = "bernoulli",
                        training_frame = train,
                        nfolds = 5,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE)

rf1 <- h2o.randomForest(x = x, y = y, #distribution = "bernoulli",
                        training_frame = train,
 
 models <- list(glm1, gbm1, dl1, rf1)
metalearner <- "h2o.glm.wrapper"

stack <- h2o.stack(models,  #models must a list of (supervised) H2O models saved using with keep_levelone_data = TRUE and identical fold_column
                   response_frame = train[,y],
                   metalearner = metalearner, 
                   seed = 1,
                   keep_levelone_data = TRUE)


# Compute test set performance:
perf <- h2o.ensemble_performance(stack, newdata = test)

# The "perf" object has a print method, so we can print results (for the default metric) by simply typing: perf
perf 
# H2O Ensemble Performance on <newdata>:
# ----------------
# Family: binomial

# Ensemble performance (AUC): 0.78122076490864

# Base learner performance:
#   learner       AUC
# 1             GLM_model_R_1457489971901_1 0.6823032
# 2            GBM_model_R_1457489971901_19 0.7780807
# 3 DeepLearning_model_R_1457489971901_1124 0.6980040
# 4          DRF_model_R_1457489971901_1153 0.7546005

# For h2oEnsemble v0.1.6 and below, you must generate the performance metrics yourself.
# An example of how to do this is below:

# Generate predictions on the test set
pp <- predict(stack, test)
predictions <- as.data.frame(pp$pred)[,3]  #third column, p1 is P(Y==1)
labels <- as.data.frame(test[,y])[,1]

# Ensemble test AUC 
cvAUC::AUC(predictions = predictions, labels = labels)
# 0.7812278

# Base learner test AUC (for comparison)
L <- length(models)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pp$basepred)[,l], labels = labels)) 
data.frame(learner=names(stack$basefits), auc)
# learner       auc
# 1             GLM_model_R_1457489971901_1 0.6823090
# 2            GBM_model_R_1457489971901_19 0.7780352
# 3 DeepLearning_model_R_1457489971901_1124 0.6979997
# 4          DRF_model_R_1457489971901_1153 0.7545723
