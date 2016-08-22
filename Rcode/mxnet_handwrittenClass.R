#  https://www.kaggle.com/c/digit-recognizer
require(mxnet)
require(data.table)
setwd("./Documents/R/Rcode//data/")

train <- fread('handdc_train.csv', header=TRUE)
test <- fread('handdc_test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)
View(head(train))

train.x <- train[,-1]
train.y <- train[,1]

# The greyscale of each image falls in the range [0, 255], we can linearly transform it into [0,1] by
train.x <- t(train.x/255)
test <- t(test/255)

table(train.y)


# network configuration
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

# In mxnet, we use its own data type symbol to configure the network. 
# data <- mx.symbol.Variable("data") use data to represent the input data, i.e. the input layer.
# Then we set the first hidden layer by fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128). This layer has data as the input, its name and the number of hidden neurons.
# The activation is set by act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu"). The activation function takes the output from the first hidden layer fc1.
# The second hidden layer takes the result from act1 as the input, with its name as "fc2" and the number of hidden neurons as 64.
# the second activation is almost the same as act1, except we have a different input source and name.
# Here comes the output layer. Since there's only 10 digits, we set the number of neurons to 10.
# Finally we set the activation to softmax to get a probabilistic prediction.

devices <- mx.cpu()

model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# Predict
preds <- predict(model, test)
dim(preds)

pred.label <- max.col(t(preds)) - 1
table(pred.label)

submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)



#  LeNet
# LeNet. It is proposed by Yann LeCun to recognize handwritten digits. Now we are going to demonstrate how to construct and train an LeNet in mxnet


# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# reshape the matrices into arrays:
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

# Next we are going to compare the training speed on different devices, so the definition of the devices goes first:
n.gpu <- 1
device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})

tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=1, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

#  with GPU
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.gpu, num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# predict
preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)



