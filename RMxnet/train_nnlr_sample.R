library(mxnet)

# data prepare
data(BostonHousing, package="mlbench")
train.ind = seq(1, 506, 3)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]

# define the network
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

# the data should be 4D in batch-num_filter-y-x
# cn1 <- mx.symbol.Convolution(data=fc1, kernel=c(2,2), num_filter = 50)
lro <- mx.symbol.LinearRegressionOutput(fc1)

# train model
mx.set.seed(0)
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)

# see the computation graph
graph.viz(model$symbol$as.json())

# save model
mx.model.save(model, './trainedmodel/nn_lr', iteration = 10)


# load pretrained model
model <- mx.model.load('./trainedmodel/nn_lr', iteration = 10)
