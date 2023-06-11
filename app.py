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

st.set_page_config(page_title="Cloud Classy",page_icon="☁️",layout="wide",initial_sidebar_state="expanded")
model = load_learner('Cloud_resnet50_fastai.pkl',cpu=True) # load model

st.title("Cloud Classification (Cloud Classy) มามะมาแยกเมฆกัน") #Title
st.markdown('"Cloud _Classy" is a project that will help you identify a cloud type from the image you upload.') #information
st.markdown("Please upload your image of cloud or use the sample images on the left sidebar.") #information
st.sidebar.image('./logo.jpg')
st.sidebar.markdown("**ถ้าขี้เกียจหรือไม่สะดวกหารูปก็เลือกข้างล่างนี้เลยยย**\n\n\nV\nV\nV\nV\nV\nV\nV\nV\nV\nV")

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
c_type = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Contrails', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']
im_predicted = model.predict(img) #predict model
c_name = im_predicted[0]
ts_prob = im_predicted[2]
prob = torch.sort(ts_prob, descending=True)
m_prob = prob[0][0]

if c_name in c_type:
     st.success(f"This cloud is **{c_name}**  with the probability of **{m_prob*100:.02f}**%") #result displays
     st.snow()
     if c_name == c_type[0]:
         st.title(f"**{c_name}** \n **อัลโตคิวมูลัส (เมฆชั้นกลาง)** :\n")
         st.markdown(' ')
         st.markdown('เมฆนี้ไม่ทำให้เกิดน้ำฟ้าบ่งบอกว่าลักษณะอากาศดีโดยเฉพาะหลังพายุฝนมีลักษณะเป็นก้อนเล็กๆเป็นหย่อมแผ่นหรือชั้นคล้ายเกล็ดก้อนกลมหรือม้วนมีทั้งสีเทาหรือทั้งสองสีบางครั้งอาจเห็นคล้ายปุยหรือฝ้าซึ่งประกอบด้วยละอองน้ำจำนวนมากมักทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลด.')
         st.markdown('เป็นเมฆชั้นกลาง: เกิดขึ้นที่ระดับความสูง 2-6 กิโลเมตร ประกอบด้วยผลึกน้ำแข็งและอนุภาคน้ำ เพราะที่ระดับนี้มีอุณหภูมิไม่ต่ำพอที่จะเป็นผลึก.')
     elif c_name == c_type[1]:
         st.title(f"**{c_name}** \n **อัลโตสเตรตัส (เมฆชั้นกลาง)** :\n")
         st.markdown(' ')
         st.markdown('เมื่อเห็นเมฆชนิดนี้อาจทำให้เกิดฝน หิมะ หรือลูกปรายน้ำแข็ง โดยจะมีลักษณะเป็นปุย แผ่น หรือเนื้อเดียวกัน พบได้ทั้งสีเทาหรือสีฟ้าอ่อน อาจทำให้เมื่อมองดวงอาทิตย์จะเห็นได้แบบสลัวๆ เหมือนมองผ่านกระจกฝ้า')
         st.markdown('เป็นเมฆชั้นกลาง: เกิดขึ้นที่ระดับความสูง 2-6 กิโลเมตร ประกอบด้วยผลึกน้ำแข็งและอนุภาคน้ำ เพราะที่ระดับนี้มีอุณหภูมิไม่ต่ำพอที่จะเป็นผลึก.')
     elif c_name == c_type[2]:
         st.title(f"**{c_name}** \n เมฆเซอโรคิวมูลัส หรือซีร์โรคิวมูลัส (เมฆชั้นสูง):\n")
         st.markdown(' ')
         st.markdown('เมฆลักษณะนี้ก็บ่งบอกลักษณะอากาศดีเช่นกันลักษณะเป็นหย่อม,แผ่น,หรือชั้นบางๆ สีขาว คล้ายเมฆก้อนเล็กๆ มีทั้งที่อยู่คิดกันและแยกกันเรียงตัวอย่างเป็นระเบียบเมฆชนิดนี้อาจทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลดและปรากฏการณ์แถบสีหรือรุ้ง')
         st.markdown('ประกอบด้วยผลึกน้ำแข็ง เป็นเมฆที่เป็นระลอกคลื่นหรือก้อนกลมๆ เป็นแถวๆ ลักษณะคล้ายเกล็ดปลา เรียงกันเป็นระเบียบ ฐานเมฆสูงเฉลี่ยประมาณ 7,000 เมตร')
         st.markdown('เป็นเมฆชั้นสูง: ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
     elif c_name == c_type[3]:
         st.title(f"**{c_name}** \n เมฆเซอโรสเตรตัส หรือซีร์โรสเตรตัส (เมฆชั้นสูง):\n")
         st.markdown(' ')
         st.markdown('เมฆนี้ไม่ทำให้เกิดน้ำฟ้าบ่งบอกว่าลักษณะอากาศดีโดยเฉพาะหลังพายุฝนมีลักษณะเป็นก้อนเล็กๆเป็นหย่อมแผ่นหรือชั้นคล้ายเกล็ดก้อนกลมหรือม้วนมีทั้งสีเทาหรือทั้งสองสีบางครั้งอาจเห็นคล้ายปุยหรือฝ้าซึ่งประกอบด้วยละอองน้ำจำนวนมากมักทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลด.')
         st.markdown('ประกอบด้วยผลึกน้ำแข็ง เป็นเมฆที่เป็นระลอกคลื่นหรือก้อนกลมๆ เป็นแถวๆ ลักษณะคล้ายเกล็ดปลา เรียงกันเป็นระเบียบ ฐานเมฆสูงเฉลี่ยประมาณ 7,000 เมตร')
         st.markdown('เป็นเมฆชั้นสูง: ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
     elif c_name == c_type[4]:
         st.title(f"**{c_name}** \n เมฆเซอรัส หรือซีร์รัส (เมฆชั้นสูง):\n")
         st.markdown(' ')
         st.markdown('หากเห็นเมฆประเภทนี้บ่งบอกได้เลยว่าวันนั้นลักษณะอากาศดี  โดยลักษณะเป็นเส้นใยละเอียดสีขาว คล้ายปุยขนสัตว์หรือเหลือมเป็นมันเงา เกิดขึ้นเป็นหย่อมหรือแถบก็ได้ แต่จะไม่มีเงาเมฆ เมฆชนิดนี้อาจทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ แต่ยังไม่เต็มวง.')
         st.markdown('ประกอบด้วยผลึกน้ำแข็ง มีลักษณะเป็นแผ่นบางสีขาวเจิดจ้า หรือสีเทาอ่อน แสงอาทิตย์สามารถส่องผ่านได้ดี มีรูปทรงอยู่หลายแบบ เช่น เป็นฝอยคล้ายขนนกบางๆ หรือขนสัตว์ หรือเป็นทางยาว ฐานเมฆสูงเฉลี่ย 10,000 เมตร')
         st.markdown('เป็นเมฆชั้นสูง: ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
     elif c_name == c_type[5]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')
     elif c_name == c_type[6]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')
     elif c_name == c_type[7]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')
     elif c_name == c_type[8]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')
     elif c_name == c_type[9]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')
     elif c_name == c_type[10]:
         st.title(f"**{c_name}** \n :\n")
         st.markdown(' ')
         st.markdown('.')




else:
     st.error(f"Sorry, pls take another image") #result display
st.markdown(' ')
st.markdown('________________________________________________')
st.markdown(' ')
st.markdown('Link ต่างๆที่เกี่ยวข้อง')
st.markdown('Medium : https://medium.com/@Chinochi/cloud-type-classification-cloud-class-ระบบแยกประเภทของเมฆ-6c5233f1ab8')
st.markdown('GitHub : https://github.com/Mhapong/cloud_classify')
st.title('**ขอบคุณพี่ๆทีมAI Builders ที่ช่วยพัฒนาผมจากคนที่ทำอะไรไม่เป็นสักอย่างรู้สึกไม่มั่นใจกับตัวเองกลายมาเป็นเป็นคนที่สามารถทำชิ้นงานชิ้นแรกได้และชิ้นงานจะเป็นก้าวเล็กแต่สำคัญในชีวิตของผม**')
st.markdown('ขอบคุณเว็บไซต์ที่ให้ความรู้')
st.markdown('ทรูปลูกปัญญา : https://www.trueplookpanya.com/learning/detail/33793')
st.markdown('กรุงเทพธุระกิจ : https://www.bangkokbiznews.com/lifestyle/892728')