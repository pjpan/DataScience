require(mxnet)
require(imager)
# The pre-trained Inception-BatchNorm network can be downloaded from this link
# This model gives the recent state-of-art prediction accuracy on image net dataset
model = mx.model.load("Inception/Inception_BN", iteration=39)
# We also need to load in the mean image, which is used for preprocessing using mx.nd.load.
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
# Load and Preprocess the Image
# Now we are ready to classify a real image. In this example, we simply take the parrots image
# from imager. But you can always change it to other images. Firstly we will test it on a photo of Mt. Baker in north WA.
# 
# Load and plot the image:
im <- load.image("../pics/Baker_Lake.jpg")
plot(im)

# Before feeding the image to the deep net, we need to do some preprocessing
# to make the image fit in the input requirement of deepnet. The preprocessing
# includes cropping, and substraction of the mean.
# Because mxnet is deeply integerated with R, we can do all the processing in R function.

preproc.image <-function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  yy <- floor((shape[1] - short.edge) / 2) + 1
  yend <- yy + short.edge - 1
  xx <- floor((shape[2] - short.edge) / 2) + 1
  xend <- xx + short.edge - 1
  croped <- im[yy:yend, xx:xend,,]
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized)
  dim(arr) = c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

normed <- preproc.image(im, mean.img)
# Classify the Image
# Now we are ready to classify the image! We can use the predict function
# to get the probability over classes.

prob <- predict(model, X=normed)
dim(prob)
## [1] 1000    1
# As you can see prob is a 1000 times 1 array, which gives the probability
# over the 1000 image classes of the input.
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
max.idx
## [1] 981 971 980 673 975

synsets <- readLines("Inception/synset.txt")
print(paste0("Predicted Top-classes: ", synsets[max.idx]))

# SAMPLE2
im <- load.image("pics/Vancouver.jpg")
plot(im)
normed <- preproc.image(im, mean.img)
prob <- predict(model, X=normed)
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
print(paste0("Predicted Top-classes: ", synsets[max.idx]))


# sample3
im <- load.image("Pics/Switzerland.jpg")
plot(im)

normed <- preproc.image(im, mean.img)
prob <- predict(model, X=normed)
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
print(paste0("Predicted Top-classes: ", synsets[max.idx]))


