# mxnet notes
### how to transform image file into MX format

#### tools
- [im2rec.py](https://github.com/dmlc/mxnet/tree/e7514fe1b3265aaf15870b124bb6ed0edd82fa76/tools)
>transform image file to mx format type.Image RecordIO
- image.lst:configuration
> integer_image_index \t label_index \t path_to_image

#### samples
- binary Labels
```
./bin/im2rec image.lst image_root_dir output.bin resize=256
```
- image.list sample
```
895099  464     n04467665_17283.JPEG
10025081        412     ILSVRC2010_val_00025082.JPEG
74181   789     n01915811_2739.JPEG
10035553        859     ILSVRC2010_val_00035554.JPEG
10048727        929     ILSVRC2010_val_00048728.JPEG
94028   924     n01980166_4956.JPEG
1080682 650     n11807979_571.JPEG
972457  633     n07723039_1627.JPEG
```

- Multiple Labels
```
integer_image_index \t label_1 \t label_2 \t label_3 \t label_4 \t path_to_image
```
```
./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4
```

- mx.io.ImageRecordIter
```
dataiter = mx.io.ImageRecordIter(
  path_imgrec="data/cifar/train.rec",
  data_shape=(3,28,28),
  path_imglist="data/cifar/image.lst",
  label_width=4              # Multiple Labels
)
```

## how to deal with picture
