import streamlit as st
from PIL import Image
import os
import json
import numpy as np
import pandas as pd
from fastai.vision.all import load_learner
from fastai.vision.all import PILImage
from fastai.vision.all import Resize
from fastai.vision.all import *
from fastai.vision.widgets import *
import pathlib

c_type = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Contrails', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']
model = load_learner("./Cloud_resnet50_fastai.pkl") # load model

st.title("Cloud _Classy") #Title
st.markdown('"Cloud _CLassy" is a project that will help you identify a cloud type from the image you upload.') #information
st.markdown("Please upload your image of cloud or use the sample images on the left sidebar.") #information

sample_path = ("./sample_images") #folder sameple images
file_name = os.listdir(sample_path) #file name from folder
sample_image = st.sidebar.selectbox(   #create selectbox sidebar
    'Sample images',
    (file_name))

file = st.file_uploader("Upload your image") #upload file
file = get_transforms()
if file is None:
    img = PILImage.create(os.path.join(sample_path, sample_image))
    st.title("Here is the sample image") #display sample image
    st.image(img)

else:
    img = PILImage.create(file)
    st.title("Here is the image you've selected") #display selected image
    st.image(img)

im_predicted = model.predict(img) #predict model

if a in c_type:
    st.success(f"This cloud is **{a}**  with the probability of **{c[b]*100:.02f}**%") #result displays
    st.balloons()

else:
    st.success(f"This cloud is **{a}**  with the probability of **{c[b]*100:.02f}**%") #result display

st.markdown('Link ต่างๆที่เกี่ยวข้อง')
st.markdown('Medium : https://medium.com/@Chinochi/cloud-type-classification-cloud-class-ระบบแยกประเภทของเมฆ-6c5233f1ab8')
st.markdown('GitHub : https://github.com/Mhapong/cloud_classify')