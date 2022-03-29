import cv2
import tensorflow as tf 
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")
                          ])



#set optimizers ,loss functions, and metrics
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# epoch: how many times the model will train with this data (each image)
model.fit(x_train, y_train, epochs = 8)

# score model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

model.save('saved_models/model1')