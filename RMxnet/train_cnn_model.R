library(FNN)
library(jpeg)
library(imager)
library(mxnet)

work_path <- c("D:/PPT/图片/")
nrow = length(list.files(work_path))
im <- NULL
# i <- 1

PreprocessImage <- function(path,show_img = T){
  
  img <- load.image(path)
  
  short_edge = min(dim(img)[1:2])

  yy = (dim(img)[1] - short_edge) / 2
  xx = (dim(img)[2] - short_edge) / 2
  crop_img = imresize(as.cimg(img[(yy+1) : (yy + short_edge), (xx+1) : (xx + short_edge),,],scale=1))

  # convert to numpy.ndarray
  sample = as.array(crop_img) * 256
  # cat(dim(sample))
  # swap axes to make image from (299, 299, 1, 3) to (1, 3, 299, 299)
  sample <- permute_axes(sample,"zcxy")
  # sub mean
  normed_img = sample - 128.
  normed_img = normed_img /128.
  # print("transformed Image Shape:", dim(normed_img))
  return(normed_img)
}

# load img file
for(i in 3){
  if(list.files(work_path)[i]){
    path = paste0(work_path, list.files(work_path)[i])
    img  <- PreprocessImage(path)
  }
}

# train a cnn network
library(mxnet)

get_lenet <- function() {
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
  # tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
  # # second fullc
  # fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
  # # loss
  # lenet <- mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
  # lenet
  return(fc1)
}

# 根据channel来进行卷积,并拼成同一个数组；
get_symbols <- function(img)
{
  nchannel = dim(img)[2]
  simg = mx.symbol.Variable("img")
  skernel = mx.symbol.Variable("kernel")
  channels = mx.symbol.SliceChannel(simg, num.outputs = nchannel)
  
  conv_r = mx.symbol.Convolution(data = channels[[1]], weight = skernel
                        ,num.filter = 1, kernel = c(3, 3), pad = c(1, 1),
                        no.bias = T, stride = c(2, 2))
  
  conv_g = mx.symbol.Convolution(data=channels[[2]],weight=skernel
                                 ,num.filter = 1,kernel = c(3, 3), pad = c(1, 1),
                                 no.bias = T, stride = c(2, 2))
  
  conv_b = mx.symbol.Convolution(data=channels[[3]],weight=skernel
                                 ,num.filter = 1,kernel = c(3, 3), pad = c(1, 1),
                                 no.bias = T, stride = c(2, 2))
  
  out = mx.symbol.Concat(list(conv_r, conv_g, conv_b),num.args = 3)
  return(out)
}

# 
mx.model.FeedForward.create(symbol = get_symbols, X = img, y = "1")



























