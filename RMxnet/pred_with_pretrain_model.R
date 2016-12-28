library(mxnet)
library(imager)

# load the pretrained mxnet model
# Inception_BN-0039
# it means prefix = 'Inception_BN', iteration = '39'
# iteration should be equal to the export file.
# first para should be prefix，not a filename；
model <- mx.model.load("./inception-bn/Inception-BN", iteration = 126)
graph.viz(model$symbol$as.json())

model$arg.params

# 查看中间层
internals <- model$symbol$get.internals()
internals$outputs

# load in the mean image
mean.img <- as.array(mx.nd.load("./inception-bn/mean_224.nd")[["mean_img"]])
summary(mean.img)

# load and preprocess the image
im <- load.image(system.file("extdata/parrots.png", package = "imager"))
plot(im)

# preproce image to the uniform
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.image
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

# 
dim(mx.nd.array(normed))
# get the normaled image;
normed <- preproc.image(im, 224)
dim(normed)
# predict 
prob <- predict(model, X = normed)
dim(prob)

model$symbol$arguments

# select the top prob
max.idx <- max.col(t(prob))
max.idx

# see the very category
synsets <- readLines("./inception-bn/synset.txt")
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

# 
model$arg.params$bn_4a_double_3x3_1_gamma









