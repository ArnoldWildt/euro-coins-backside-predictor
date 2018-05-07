import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', help="Filename")
parser.add_argument('-gray', help="If Img is already gray it skips the convert", dest='is_gray',action="store_true")

args = parser.parse_args()
filename = args.file
is_gray = args.is_gray

import plaidml.keras
plaidml.keras.install_backend()

from resnet import ResnetBuilder

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imshow, imsave

from PIL import Image

import os

import numpy as np

import matplotlib.pyplot as plt

names = ["France", "Greece", "Germany","DE - Sachsen", "Spain", "Italy 2Euro", "Luxembourg", "Italy 1Euro"]

model = ResnetBuilder.build_resnet_34((200, 200, 1), 8)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy'])

path = "B:/Desktop/Projekts/Euro_Pred/pred_img/"

pred_im = Image.open(path + filename)
pred_im = np.array(pred_im)

if not is_gray:
	pred_im = rgb2gray(pred_im)

	pred_im = resize(pred_im, (200,200))

	imsave(path + "gray_"+ filename, pred_im)

	pred_im = pred_im * 256

pred_im = np.reshape(pred_im, (1,200,200,1))

model.load_weights("B:/Desktop/Projekts/Euro_Pred/saved_models/model_res_val+rmsdrop.011.h5")

print(pred_im)
output = model.predict(pred_im,verbose=1)

print(output)
print(output*100)
arg = np.argmax(output)
print(arg)
print(names[arg])
print(output[0,arg]*100, "%")