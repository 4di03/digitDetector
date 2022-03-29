import cv2
import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('saved_models/model1')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

#plt.imshow(x_test[3])

def predict(img):
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = img *255

    #print(img)
    return np.argmax(model.predict(img))



def draw(event, x, y, flag,  param):
    global run

    if event == cv2.EVENT_LBUTTONDOWN:
       run = True

    if event == cv2.EVENT_LBUTTONUP:
        run = False

    if event == cv2.EVENT_MOUSEMOVE and run:
       cv2.circle(win, (x,y), 10 , 1 , -1)

    
#create blank windows for drwawing
cv2.namedWindow('window')
cv2.setMouseCallback('window', draw)

win = np.zeros((280, 280, 1) , dtype = 'float64')
win2 = cv2.putText(np.zeros((500, 700, 3), dtype = 'float64'), 'Press s to save image for Prediction!', (50, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)

win2 = cv2.putText(win2, 'Neural Net Prediction:', (50, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)




while True:
    cv2.imshow('window', win)
    cv2.imshow('output', win2)

    k = cv2.waitKey(1)

    if k == ord('c'):
        win = np.zeros((280, 280, 1) , dtype = 'float64')

    if k == ord('s'):
        win2[120:400, 20:300] = win
        win2[50:500, 350:700] = 0
        win2 = cv2.putText(win2, str(predict(win)), (350, 390), cv2.FONT_HERSHEY_SIMPLEX, 15, (0, 0, 255), 15, cv2.LINE_AA)
        


    if k == 27:
        cv2.destroyAllWindows()
        break
    








