import plaidml.keras
plaidml.keras.install_backend()

from resnet import ResnetBuilder

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from PIL import Image

import os

import numpy as np

import matplotlib.pyplot as plt

model = ResnetBuilder.build_resnet_34((200, 200, 1), 8)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy'])

#model.compile(loss="categorical_crossentropy", optimizer="sgd")

#model.load_weights("B:/Desktop/Projekts/Euro_Pred/saved_models/model_res_new_pic.006.h5")

path = "B:/Desktop/Projekts/Euro_Pred/new_pics/output/"

buffer = []
counter = 0
count_val = 0
counter_val = 0

x_train = np.zeros([9000,200,200])
y_train = np.zeros([9000,8])
x_val = np.zeros([1000,200,200])
y_val = np.zeros([1000,8])

print("Training size: ", x_train.shape)
print("Val size: ",x_val.shape)

print("Loading Images")

for filename in os.listdir(path):

	buffer = filename.split("_")
	im = Image.open(path+filename)
	im2arr = np.array(im)
	
	count_val += 1

	if count_val == 10:
		y_val[counter_val,int(buffer[1])-1] = 1
		x_val[counter_val] = im2arr
		counter_val += 1
		count_val = 0
	else:
		y_train[counter,int(buffer[1])-1] = 1
		x_train[counter] = im2arr
		counter += 1
	

print("Done loading images")

x_test = x_train[np.random.randint(1,9000)]

plt.imshow(x_test)
plt.show()

x_train = np.reshape(x_train, (9000,200,200,1))
x_val = np.reshape(x_val, (1000,200,200,1))

x_test = x_test.reshape(1,200,200,1)


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model_res_val+rmsdrop.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint]

model.fit(x_train, y_train, batch_size=None, epochs =20, verbose=1, shuffle=True, callbacks=callbacks, validation_data=(x_val, y_val))

output = model.predict(x_test,batch_size=1,verbose=1)

print(output)
print(np.argmax(output))