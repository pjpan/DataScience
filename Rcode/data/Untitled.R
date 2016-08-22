require(mxnet)
require(imager)

require(imager)
Load the Pretrained Model
Make sure you unzip the pre-trained model in current folder. And we can use the model
loading function to load the model into R.

model = mx.model.load("Inception/Inception_BN", iteration=39)
We also need to load in the mean image, which is used for preprocessing using mx.nd.load.

mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
# Load and Preprocess the Image
# Now we are ready to classify a real image. In this example, we simply take the parrots image
# from imager. But you can always change it to other images. Firstly we will test it on a photo of Mt. Baker in north WA.
# 
# Load and plot the image:
  
  im <- load.image("Pics/MtBaker.jpg")
plot(im)
plot of chunk unnamed-chunk-5

Before feeding the image to the deep net, we need to do some preprocessing
to make the image fit in the input requirement of deepnet. The preprocessing
includes cropping, and substraction of the mean.
Because mxnet is deeply integerated with R, we can do all the processing in R function.

The preprocessing function:
  
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
We use the defined preprocessing function to get the normalized image.

normed <- preproc.image(im, mean.img)
Classify the Image
Now we are ready to classify the image! We can use the predict function
to get the probability over classes.

prob <- predict(model, X=normed)
dim(prob)
## [1] 1000    1
As you can see prob is a 1000 times 1 array, which gives the probability
over the 1000 image classes of the input.

We can extract the top-5 class index.

max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
max.idx
## [1] 981 971 980 673 975
These indices do not make too much sense. So let us see what it really represents.
We can read the names of the classes from the following file.

synsets <- readLines("Inception/synset.txt")
And let us print the corresponding lines:
  
  print(paste0("Predicted Top-classes: ", synsets[max.idx]))
## [1] "Predicted Top-classes: n09472597 volcano"      
## [2] "Predicted Top-classes: n09193705 alp"          
## [3] "Predicted Top-classes: n09468604 valley, vale" 
## [4] "Predicted Top-classes: n03792972 mountain tent"
## [5] "Predicted Top-classes: n09288635 geyser"
Mt. Baker is indeed a vocalno. We can also see the second most possible guess "alp" is also correct.

Let's see if it still does a good job on some other images. The following photo is taken in Vancouver downtown.

im <- load.image("Pics/Vancouver.jpg")
plot(im)
plot of chunk unnamed-chunk-12

normed <- preproc.image(im, mean.img)
prob <- predict(model, X=normed)
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
print(paste0("Predicted Top-classes: ", synsets[max.idx]))
## [1] "Predicted Top-classes: n09332890 lakeside, lakeshore"    
## [2] "Predicted Top-classes: n03983396 pop bottle, soda bottle"
## [3] "Predicted Top-classes: n13133613 ear, spike, capitulum"  
## [4] "Predicted Top-classes: n12144580 corn"                   
## [5] "Predicted Top-classes: n02980441 castle"
This photo is indeed taken at lakeside. One interesting guess is the fifth guess "castle". The outline of the building in the city is recognized as the battlements on a castle. We might need more pictures containing "battlements with glass windows" to teach the model about modern city.

How about this photo taken on Titlis:

im <- load.image("Pics/Switzerland.jpg")
plot(im)
plot of chunk unnamed-chunk-13

normed <- preproc.image(im, mean.img)
prob <- predict(model, X=normed)
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
print(paste0("Predicted Top-classes: ", synsets[max.idx]))
## [1] "Predicted Top-classes: n04371774 swing"                         
## [2] "Predicted Top-classes: n04275548 spider web, spider's web"      
## [3] "Predicted Top-classes: n01773549 barn spider, Araneus cavaticus"
## [4] "Predicted Top-classes: n03000684 chain saw, chainsaw"           
## [5] "Predicted Top-classes: n03888257 parachute, chute"
This time the main element is small and cannot stand out from the "noisy" background. This time the result is not perfect, but we can still find similarity between "swing" and "gondola".

Now, why don't you take a photo around and ask mxnet to tell you what is included? Have some fun!