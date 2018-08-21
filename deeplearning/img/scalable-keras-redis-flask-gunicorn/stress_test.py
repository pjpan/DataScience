#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/21/18 10:20 PM
# @Author  : PPj
# @Site    : 
# @File    : stress_test.py
from threading import Thread
import requests
import time


# initialize the keras REST API endpoint URL
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "/Users/ppj/Downloads/1.jpg"

# initialize the number of request for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 500
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
    # load the input image and construct the payload for the requests
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    # ensure the request was sucessful
    if r["success"]:
        print("[INFO] thread {} OK".format(n))

    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))

# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep((SLEEP_COUNT))

# insert a long sleep so we can wait until the server is finished
time.sleep(300)
