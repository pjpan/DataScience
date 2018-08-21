#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/21/18 1:09 AM
# @Author  : PPj
# @Site    : 
# @File    : settings.py

# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 12
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25