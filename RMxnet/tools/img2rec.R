library(mxnet)
library(imager)


list_image <- function(root, recursive, exts) {
  i <- 0
  image_list <- NULL
  if(recursive) {
    cat <- NULL
    for()
    
  } else {
    
  }

}



image_encode <- function(args, item, q_out){
  
  img <- load.image()
  
}


def image_encode(args, item, q_out):
  try:
  img = cv2.imread(os.path.join(args.root, item[1]), args.color)
except:
  print('imread error:', item[1])
return
if img is None:
  print('read none error:', item[1])
return
if args.center_crop:
  if img.shape[0] > img.shape[1]:
  margin = (img.shape[0] - img.shape[1]) / 2;
img = img[margin:margin + img.shape[1], :]
else:
  margin = (img.shape[1] - img.shape[0]) / 2;
img = img[:, margin:margin + img.shape[0]]
if args.resize:
  if img.shape[0] > img.shape[1]:
  newsize = (args.resize, img.shape[0] * args.resize / img.shape[1])
else:
  newsize = (img.shape[1] * args.resize / img.shape[0], args.resize)
img = cv2.resize(img, newsize)
if len(item) > 3 and args.pack_label:
  header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
else:
  header = mx.recordio.IRHeader(0, item[2], item[0], 0)

try:
  s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
q_out.put((s, item))
except Exception, e:
  print('pack_img error:', item[1], e)
return


