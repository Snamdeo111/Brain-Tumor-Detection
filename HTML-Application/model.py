import cv2
from tensorflow import keras
import imutils
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings("ignore")
IMG_SIZE = (224,224)
model = keras.models.load_model('VGG_model.h5')
pred_dict={0: "Tumor Not Present",1: "Tumor Present"}
file_path_list=[]
def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    img=set_name
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = add_pixels_value
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    set_new.append(new_img)
    return np.array(set_new)

def preprocess_imgs(set_name):
    set_new = []
    img = cv2.resize(
            set_name[0],
            dsize=IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
    set_new.append(preprocess_input(img))
    return np.array(set_new)

def predict(image):
    image = cv2.imread(image)
    image = crop_imgs(set_name=image)
    image = preprocess_imgs(set_name=image)
    result = model.predict(image)
    result = [1 if x > 0.5 else 0 for x in result]
    result = pred_dict.get(result[0])
    return result