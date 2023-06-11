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
import pathlib
import urllib

c_type = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Contrails', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']
model = load_learner('Cloud_resnet50_fastai.pkl',cpu=True) # load model

st.title("Cloud Classification มามะมาแยกเมฆกัน") #Title
st.markdown('"Cloud _CLassy" is a project that will help you identify a cloud type from the image you upload.') #information
st.markdown("Please upload your image of cloud or use the sample images on the left sidebar.") #information
st.sidebar.markdown("ถ้าขี้เกียจหรือไม่สะดวกหารูปก็เลือกข้างล่างนี้เลยยย\n\n\nV\nV\nV\nV\nV\nV\nV\nV\nV\nV")

sample_path = ("./sample_images") #folder sameple images
file_name = os.listdir(sample_path)
sample_image = st.sidebar.selectbox(   #create selectbox sidebar
    'Sample images',
    (file_name))

st.sidebar.markdown("⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ‧ ⁺ ✦ ⊹ . * ꙳ ✦ ⊹ ⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ‧ ⨯. ⁺ ✦ ☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩⋆✧⭒☁️⭑⋆⭒🟉☆⭑☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡☁︎︎⋆｡˚✩☁︎︎⋆｡⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ‧ ⁺ ✦ ⊹ . * ꙳ ✦ ⊹ ⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ‧ ⨯. ⁺ ✦ ☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩✩☁︎︎⋆｡˚✩☁︎︎⋆｡☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩⋆✧⭒☁️⭑⋆⭒🟉☆⭑☁︎︎⋆｡˚⋆｡˚ ☁︎ ˚｡⋆｡˚｡⋆✩☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ✦ ⋆✧⭒☁️⭑⋆⭒🟉☆⭑⊹ . * ꙳ ✦ ⊹ ⨯ . ⁺ ✦ ⊹ ꙳ ⁺ ‧ ⨯. ⁺ ✦ ☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩☁︎︎⋆｡˚✩⋆｡˚ ☁︎ ˚｡⋆｡˚｡⋆｡˚✩☁︎︎⋆｡☁︎︎⋆｡˚✩☁︎︎⋆｡˚ ☁︎ ˚｡⋆｡˚｡⋆˚✩☁︎︎⋆｡˚✩☁︎︎")

file = st.file_uploader("Upload your image") #upload file
if file is None:
    img = PILImage.create(os.path.join(sample_path, sample_image))
    st.title("Here is the sample image") #display sample image
    st.image(img)

else:
    img = PILImage.create(file)
    st.title("Here is the image you've selected") #display selected image
    st.image(img)

im_predicted = model.predict(img) #predict model
c_name = im_predicted[0]
ts_prob = im_predicted[2]
prob = torch.sort(ts_prob, descending=True)
m_prob = prob[0][0]

if c_name in c_type:
     st.success(f"This cloud is **{c_name}**  with the probability of **{m_prob*100:.02f}**%") #result displays
     st.balloons()

else:
     st.success(f"Sorry, pls take another images") #result display

st.markdown('Link ต่างๆที่เกี่ยวข้อง')
st.markdown('Medium : https://medium.com/@Chinochi/cloud-type-classification-cloud-class-ระบบแยกประเภทของเมฆ-6c5233f1ab8')
st.markdown('GitHub : https://github.com/Mhapong/cloud_classify')