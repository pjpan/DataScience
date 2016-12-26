library(mxnet)

ita = mx.io.CSVIter(
  data_csv = './dat.csv',
  data_shape =c(2,),
  label_csv = './lab.csv',
  label_shape = c(1,), 
  batch_size=1
)

# import image iter
train <- mx.io.ImageRecordIter(
  path.imgrec     = c("../data/cifar.rec"),
  batch.size      = 128,
  data.shape      = c(28, 28, 3),
  rand.crop       = TRUE,
  rand.mirror     = TRUE
)
















