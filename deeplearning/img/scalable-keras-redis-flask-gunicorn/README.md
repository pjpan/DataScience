## 整个目录结构
- helper.py 主要用来进行图像的基础处理，把图片变成base64进行存储
- run_model_server 用来启动keras的模型文件
- run_web_server 用来启动flask
- settings 用来存储配置文件
- stress_test 用来压力测试用

## 其中的一些注意点
1. run_model_server 是要先启动的，用来把模型文件常驻内存
2. run_web_server 用来启动flask，把webserver和modelserver进行区分，方便进行模型的维护

## 启动多进程和线程，可以用gunicorn进行设置，外面再用nginx进行keep alived来进行负载均衡；
```
gunicorn -w 2 -b 127.0.0.:5000 run_web_server:app
```
