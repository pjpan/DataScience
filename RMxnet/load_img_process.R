library(FNN)
library(jpeg)
library(imager)
library(mxnet)

work_path <- c("D:/PPT/图片/")
nrow = length(list.files(work_path))
im <- NULL
# i <- 1


# load the data and transform to size(1,3,244,244)
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
    path = paste0(work_path, list.files(work_path)[1])
    img  <- PreprocessImage(path)
  }
}


# img.process 2
#  mxnet neeed img format( width, height, channel, num)
im <- load.image(system.file("extdata/parrots.png", package = "imager"))
plot(im)

preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # dim(croped)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr)
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.image
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

# use the functions
normed <- preproc.image(im, mean.image = 128)
plot(normed)

# 




