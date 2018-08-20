#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/19/18 5:13 PM
# @Author  : Aries
# @Site    : 
# @File    : run_keras_server.py

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io


# initialize constants used to control image spatial
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server quenuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEPP = 0.25
CLIENT_SLEEP = 0.25

# initialize our flask application,redis server, and keras model
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None


# trans img to base64 encode
def base64_encode_image(img):
    # base64 encode the iput numpy array
    return base64.b64encode(img).decode("utf-8")


def base64_decode_image(img, dtype, shape):
    img = bytes(img, encoding="utf-8")

    # convert the string to a Numpy array using the supplied data
    # type and target shape
    img = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    img = img.reshape(shape)

    # return the decoded image
    return img


def prepare_image(image, target):
    # if the image model is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    #return the procesed image
    return image


# This function will poll for image batches from the Redis server,
# classify the images, and return the results to the client.
def classify_process():
    # load the pre-trained keras model
    # pre-trained on imagenet and provied by keras ,but you can
    # substitude in your own network just as easily
    print("* Loading model ...")
    model = ResNet50(weights="imagenet")
    print("* model Loaded")
    # continually poll for new images to classify
    while True:
        # attempt to grad a batch of images from the database
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE-1)
        imageIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                    (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))

            # check to see if the batch list is None
            if batch is None:
                batch = image

            # otherwise,stack the ata
            else:
                batch = np.vstack([batch, image])

            # update the list of image IDs
            imageIDs.append(q["id"])

            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                # classify the batch
                print("* Batch size: {}".format(batch.shape))
                preds = model.predict(batch)
                results = imagenet_utils.decode_predictions(preds)
                print(results)

            # loop over the image IDs and their corresponding set of result from our model
                for (imageID, resultSet) in zip(imageIDs, results):
                    output = []

                    # loop over the results and add them to the list of output predictions
                    for (imagenetID, label, prob) in resultSet:
                        r = {"label": label, "probability": float(prob)}
                        output.append(r)

                    # store the output prediction in the database,using the image id as the key so we can fetch the results
                    db.set(imageID, json.dumps(output))

                # remove the set of images from our queue
                db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

            # sleep for a small amount
            time.sleep(SERVER_SLEPP)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # ensure our numpy array is C-contigurs as well, otherwise we won't be able to serialize it
            image = image.copy(order="C")

        # generate an ID for the classification then add the classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                # check to see if our model has classified the input image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # delete the result from the database and break from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance to classify the input image
                time.sleep(CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a separate
    # thread than the one used for main classification
    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    # start the web server
    print("* Starting web service")
    app.run()

