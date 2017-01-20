# mxnet notes
### transform imagefiles
#### tools
- [im2rec.py](https://github.com/dmlc/mxnet/tree/e7514fe1b3265aaf15870b124bb6ed0edd82fa76/tools)

针对原始数据，两种存储方法

1、第一种是把同一个类别的图片放在同一个文件夹内，然后用文件夹来进行区别；
- 用im2rec先生成img.lst,参数定义--list true 表示生成的是img.lst
- 用im2rec生成rec文件；
- im2rec.py  prefix 数据源 其他参数，前两个是必选的；

2、把所有的图片文件放在一个大文件夹内；
- 自己手动生成lst文件；
- 用im2rec生成rec文件；

第一步生成的img.lst的语法如下：
### make image list
```
python im2rec.py /data1/ILSVRC2012/train/  /data1/ILSVRC2012/train_raw --train-ratio=0.7 --test-ratio=0.2 --list=True --recursive=True --resize=227
python ~/mxnet/tools/im2rec.py --list True --recursive True caltech-256-60-train caltech_256_train_60/
python ~/mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 caltech-256-60-val 256_ObjectCategories/
```

output:
> /data1/ILSVRC2012/train_test.lst
/data1/ILSVRC2012/train_train.lst
/data1/ILSVRC2012/train_val.lst

### create rec
```
python im2rec.py /data1/ILSVRC2012/train/  /data1/ILSVRC2012/train_raw --num-thread=28  --recursive=True --resize=227
```

#### Q&A
-  label_index should be integer not a string；
integer_image_index \t label_index \t path_to_image
