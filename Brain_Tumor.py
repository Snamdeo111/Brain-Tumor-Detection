from tkinter import *
import logging
logging.getLogger('tensorflow').disabled = True
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter.filedialog import askopenfilename
import sys
from tensorflow import keras
from tkinter import PhotoImage
from PIL import Image, ImageTk
import imutils
from tensorflow.keras.applications.vgg16 import preprocess_input
IMG_SIZE = (224,224)
model = keras.models.load_model('VGG_model.h5')
pred_dict={0: "Tumor Not Present",1: "Tumor Present"}
file_path=''
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
def display_image(image,label):
    image=cv2.imread(image)
    image=crop_imgs(set_name=image)
    imag=preprocess_imgs(set_name=image)
    result=model.predict(imag)
    result = [1 if x>0.5 else 0 for x in result]
    result=pred_dict.get(result[0])
    label.configure(text=result)

def open_file():
    global file_path
    image_formats= [("JPEG", "*.jpg"),("PNG", "*.png"),("JPEG", "*.jpeg")]
    file_path = askopenfilename(filetypes=image_formats, initialdir="/", title='Please Select A Picture To Analyze')
    img = cv2.imread(file_path)
    img = cv2.resize(
            img,
            dsize=IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
    image = Image.fromarray(img)
    image=image.resize((630,630))
    image = ImageTk.PhotoImage(image=image) 
    label1.configure(image=image,height=700,width=700)
    label1.image=image

root = Tk()
root.title("Brain Tumor Prediction")
root.config(bg="white")
img1=PhotoImage(r"Blank.jpg")

proj=ttk.Notebook(root)
proj.pack()

frame1=Frame(proj)
frame1.pack(fill='both',expand=1)
proj.add(frame1,text="Image To Be Predicted")


label1=Label(frame1,image=img1,height=650,width=650,bg="white")
label1.pack(pady=10)
btn = Button(frame1, text ='Open', command = lambda:open_file(),font=("Bookman old style",(12),"bold"),fg="white",bg="black")
btn.pack(pady=10)
btn1 = Button(frame1, text ='Predict', command = lambda:display_image(file_path,label2),font=("Bookman old style",(12),"bold"),fg="white",bg="black")
btn1.pack(pady=10)
label2=Label(frame1,text="Prediction",font=("ariel",(20),"bold"),fg="black",bg="white")
label2.pack(pady=10)

root.mainloop()
