import cv2
import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


def predict(img):
    return 1





def draw(event, x, y, flag,  param):
    global run

    if event == cv2.EVENT_LBUTTONDOWN:
       run = True

    if event == cv2.EVENT_LBUTTONUP:
        run = False

    if event == cv2.EVENT_MOUSEMOVE and run:
       cv2.circle(win, (x,y), 5 , 0 , -1)

    
#create blank windows for drwawing
cv2.namedWindow('window')
cv2.setMouseCallback('window', draw)

win = np.ones((280, 280, 1) , dtype = 'float64')
win2 = cv2.putText(np.zeros((500, 700, 3), dtype = 'float64'), 'Press s to save image for Prediction!', (50, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)

win2 = cv2.putText(win2, 'Neural Net Prediction:', (50, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)




while True:
    cv2.imshow('window', win)
    cv2.imshow('output', win2)

    k = cv2.waitKey(1)

    if k == ord('c'):
        win = np.ones((280, 280, 1) , dtype = 'float64')

    if k == ord('s'):
        win2[120:400, 20:300] = win
        win2 = cv2.putText(win2, str(predict(win)), (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 15, (0, 0, 255), 15, cv2.LINE_AA)


    if k == 27:
        cv2.destroyAllWindows()
        break
    








