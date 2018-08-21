#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/19/18 10:50 PM
# @Author  : PPj
# @Site    : 
# @File    : simple_samples.py

import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"
# KERAS_REST_API_URL = "https://www.baidu.com/"
IMAGE_PATH = "/Users/ppj/Downloads/1.jpg"

# load the input image and construct the payload for the requests
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the prediction and display them
    for(i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
              result["probability"]))

# otherwise,the requests failed
else:
    print("Request failed")