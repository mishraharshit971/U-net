import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, flash, url_for, jsonify
import matplotlib.pyplot as plt
import numpy as np
import h5py
import PIL
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input , decode_predictions
import cv2


MODEL_ARCHITECTURE = "./model.json"
MODEL_WEIGHTS = "./Electron Microscopy_segmentation2.h5"

def getPrediction(filename):
    
    #json_file = open(MODEL_ARCHITECTURE)
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model = model_from_json(loaded_model_json)
    #model.load_weights(MODEL_WEIGHTS)

    model = tf.keras.models.load_model(MODEL_WEIGHTS)


#PREPROCESSING STEPOS
    #IMG = image.load_img(''+filename)
    #print(type(IMG))
    
    #IMG_ = IMG.resize((500, 500))
    #IMG_ = np.asarray(IMG_)
    #IMG_ = np.true_divide(IMG_, 255)

    img=cv2.imread(''+filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(500,500))
    img=np.array(img)
    img=img* 1/255.
    print(type(img))
    
    image1 = np.expand_dims(img,axis=0)
    
    
    print(model)

    
    result = model.predict(image1)
    result = np.uint8(result>=.22)

    result = np.squeeze(result,axis=0)

    result = np.squeeze(result,axis=-1)
    print(result)

    plt.figure()
    plt.imshow(result)
    plt.savefig('./static/result.png')

    #Image.fromarray(result).save('./static/result.png')
    #output="done"
    #return jsonify(output)


    #return label[1], label[2]*100