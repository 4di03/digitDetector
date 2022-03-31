import cv2
from cv2 import INTER_NEAREST
from cv2 import INTER_LINEAR
from cv2 import INTER_LANCZOS4
from cv2 import INTER_CUBIC
from cv2 import INTER_AREA
import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
print(cv2.__version__)
model = tf.keras.models.load_model('saved_models/model1')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# plt.figure()
# plt.imshow(x_test[3], cmap = 'binary')

# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

#plt.imshow(x_test[3])

def predict(img):
#INTER_LINEAR GOOD
    img = cv2.resize(img, (28, 28), interpolation=INTER_LINEAR)
    img = img *255


    # plt.figure()
    # plt.imshow(img, cmap = 'binary')
    # plt.show()
    img = np.expand_dims(img, axis=0)


    #print(img)
    return np.argmax(model.predict(img))

run = False
def draw(event, x, y, flag,  param):
    global run

    if event == cv2.EVENT_LBUTTONDOWN:
       run = True

    if event == cv2.EVENT_LBUTTONUP:
        run = False

    if event == cv2.EVENT_MOUSEMOVE:
       if run == True:
        cv2.circle(win, (x,y), 10 , 1 , -1)

    
#create blank windows for drwawing
cv2.namedWindow('Draw Here!')
cv2.setMouseCallback('Draw Here!', draw)

win = np.zeros((280, 280, 1) , dtype = 'float64')
win2 = cv2.putText(np.zeros((500, 700, 3), dtype = 'float64'), 'Press s to save image for Prediction!', (50, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)

win2 = cv2.putText(win2, 'Neural Net Prediction:', (50, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)




while True:
    cv2.imshow('Draw Here!', win)
    cv2.imshow('output', win2)

    k = cv2.waitKey(1)

    if k == ord('c'):
        win = np.zeros((280, 280, 1) , dtype = 'float64')

    if k == ord('s'):
        win2[120:400, 20:300] = win
        win2[100:500, 350:700] = 0
     
        win2 = cv2.putText(win2, str(predict(win)), (350, 390), cv2.FONT_HERSHEY_SIMPLEX, 13, (0, 0, 255), 13, cv2.LINE_AA)
        


    if k == 27:
        cv2.destroyAllWindows()
        break
    








