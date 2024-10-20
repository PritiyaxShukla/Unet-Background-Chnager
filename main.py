import tensorflow as tf 
import keras 
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import cv2
from PIL import Image

def preprocess(image_path):
    # Read and preprocess the image
    image = cv2.cvtColor(np.array(image_path) ,  cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image, (256, 256))  # Resize to match the input size expected by the model
    
    # Normalize the image to [0, 1]
    image_normalised = image_resize.astype(np.float32) / 255.0
    
    # Add batch dimension (for prediction)
    image_normalised = np.expand_dims(image_normalised, axis=0)
    
    return  image_normalised , image_resize

def bg_changer(image_pred , image , bg):
    bg_resize = cv2.resize(cv2.cvtColor(np.array(bg) , cv2.COLOR_BGR2RGB), (256, 256))
    binary_mask = (image_pred > 0.9).astype(np.uint8)
    final_image = np.where(binary_mask[:,:,None]==1 , image.astype(np.float32), bg_resize.astype(np.float32))
    final_image = np.clip(final_image , 0 , 255).astype(np.uint8)
    return bg , binary_mask, final_image 

def make_prediction(image , bg):
    model = "Binary_Unet_Model.h5"
    loaded_model = load_model(model)
    processed_image , resize_image = preprocess(image)
    image_pred = loaded_model.predict(processed_image)[0,:,:,0]
    if image_pred.max() != image_pred.min():
        image_pred = (image_pred - image_pred.min()) / (image_pred.max() - image_pred.min())
    bground , mask , final =  bg_changer(image_pred , resize_image , bg)
    return bground , image_pred, final



st.title("Background Remover and changer Using UNET Model")
image_file = st.file_uploader("Upload a Image" , type=["png" , "jpg" ,"jpeg"])
bg_file = st.file_uploader("Upload the Background Image" , type = ["png","jpg" , "jpeg"])

if image_file is not None and bg_file:
    
    
    uploaded_image = Image.open(image_file)
    uploaded_bg = Image.open(bg_file)
    bground , mask , final = make_prediction(uploaded_image , uploaded_bg)
    st.image(uploaded_image , caption=''  , use_column_width=True)
    st.markdown("<h3 style='text-align: center;'>Uploaded Image</h3>", unsafe_allow_html=True)  # Custom caption
    st.image(bground , caption = '' , use_column_width=True )
    st.markdown("<h3 style = 'text-align: center;'>Uploaded Background</h3>" , unsafe_allow_html = True)
    st.image(mask , caption = '' , use_column_width=True)
    st.markdown("<h3 style = 'text-align: center;'>Predicted Mask </h3>" , unsafe_allow_html=True)
    st.image(cv2.cvtColor(final , cv2.COLOR_BGR2RGB) , use_column_width=True)
    st.markdown("<h3 style = 'text-align: center:'> Final Background Changed Image </h3>" , unsafe_allow_html=True)

else:
    st.write("Please upload both an image and a background.")