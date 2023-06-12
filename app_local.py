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
from fastai.vision.data import ImageDataLoaders
import pathlib
import urllib

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# dblock = DataBlock(
#     blocks=(ImageBlock, CategoryBlock), #x - image ; y - single class
#     get_items=get_image_files, #get image from selected folder (path) ; return list of pic
#     splitter=GrandparentSplitter(train_name = 'train',valid_name='valid'), #use parent folder as train-valid split
#     get_y=parent_label, #use parent folder as label 
#     item_tfms=Resize(512, method=ResizeMethod.Squish), # Resize image to same size using Squish
#     batch_tfms=aug_transforms(size=512, flip_vert=False, pad_mode=PadMode.Reflection, max_lighting=0.2, p_lighting=0.75 )
#     )
# dls = dblock.dataloaders(r'C:\Users\Msi\Downloads\Added_Cloud_Pic',shuffle=True)

# st.set_page_config(page_title="Cloud Classy",page_icon="☁️",layout="wide",initial_sidebar_state="expanded")
# modelPath = Path('./Cloud_resnet50_fastai.pkl')
# empty_data = ImageDataLoaders.from_folder(modelPath)
# learn = create_cnn(empty_data,model.resnet50)
# learner = cnn_learner(dls, resnet50)
# model = learner.load(r'C:\Users\Msi\Documents\GitHub\cloud_classify\Cloud_resnet50_fastai.pkl') # load model
model = load_learner('Cloud_resnet50_fastai.pkl',cpu=True) # load model

st.title("**Cloud Classification (Cloud Classy) มามะมาแยกเมฆกัน**") #Title
st.subheader('"Cloud _Classy" is a project that will help you identify a cloud type from the image you upload.') #information
st.markdown("Please upload your image of cloud or use the sample images on the left sidebar.") #information
st.sidebar.image('./logo.png')
st.sidebar.markdown("**ถ้าขี้เกียจหรือไม่สะดวกหารูปก็เลือกข้างล่างนี้เลยน้าบบ**\n\n\nV\nV\nV\nV\nV\nV\nV\nV\nV\nV")

sample_path = ("./sample_images") #folder sameple images
file_name = os.listdir(sample_path)
sample_image = st.sidebar.selectbox(   #create selectbox sidebar
    'Sample images',
    (file_name))

st.sidebar.title("**ส่งข้อเสนอแนะที่นี่เลย~:**")
st.sidebar.markdown("**Google form:**")
st.sidebar.markdown("https://forms.gle/dYj3TJ1mExdPRNQt8")
st.sidebar.markdown("-----------------")
st.sidebar.markdown("**Contact:**")
st.sidebar.markdown("Email: mhapongg@gmail.com")
st.sidebar.markdown("-----------------")
file = st.file_uploader("Upload your image:") #upload file
if file is None:
    img = PILImage.create(os.path.join(sample_path, sample_image))
    st.markdown('\n')
    st.title("Here is the sample image:") #display sample image
    st.image(img)

else:
    img = PILImage.create(file)
    st.title("Here is the image you've selected:") #display selected image
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
         st.title(f"**{c_name}** \n ")
         st.header('**อัลโตคิวมูลัส (เมฆชั้นกลาง):**')
         st.markdown('**เป็นเมฆชั้นกลาง:** เกิดขึ้นที่ระดับความสูง 2-6 กิโลเมตร ประกอบด้วยผลึกน้ำแข็งและอนุภาคน้ำ เพราะที่ระดับนี้มีอุณหภูมิไม่ต่ำพอที่จะเป็นผลึก.')
         st.markdown('เมฆนี้ไม่ทำให้เกิดน้ำฟ้าบ่งบอกว่าลักษณะอากาศดีโดยเฉพาะหลังพายุฝนมีลักษณะเป็นก้อนเล็กๆเป็นหย่อมแผ่นหรือชั้นคล้ายเกล็ดก้อนกลมหรือม้วนมีทั้งสีเทาหรือทั้งสองสีบางครั้งอาจเห็นคล้ายปุยหรือฝ้าซึ่งประกอบด้วยละอองน้ำจำนวนมากมักทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลด.')

     elif c_name == c_type[1]:
         st.title(f"**{c_name}** \n ")
         st.header('**อัลโตสเตรตัส (เมฆชั้นกลาง):**')
         st.markdown('**เป็นเมฆชั้นกลาง:** เกิดขึ้นที่ระดับความสูง 2-6 กิโลเมตร ประกอบด้วยผลึกน้ำแข็งและอนุภาคน้ำ เพราะที่ระดับนี้มีอุณหภูมิไม่ต่ำพอที่จะเป็นผลึก.')
         st.markdown('เมื่อเห็นเมฆชนิดนี้อาจทำให้เกิดฝน หิมะ หรือลูกปรายน้ำแข็ง โดยจะมีลักษณะเป็นปุย แผ่น หรือเนื้อเดียวกัน พบได้ทั้งสีเทาหรือสีฟ้าอ่อน อาจทำให้เมื่อมองดวงอาทิตย์จะเห็นได้แบบสลัวๆ เหมือนมองผ่านกระจกฝ้า')

     elif c_name == c_type[2]:
         st.title(f"**{c_name}** \n ")
         st.header('**เมฆเซอโรคิวมูลัส หรือซีร์โรคิวมูลัส (เมฆชั้นสูง):**')
         st.markdown('**เป็นเมฆชั้นสูง:** ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
         st.markdown('เมฆลักษณะนี้ก็บ่งบอกลักษณะอากาศดีเช่นกันลักษณะเป็นหย่อม,แผ่น,หรือชั้นบางๆ สีขาว คล้ายเมฆก้อนเล็กๆ มีทั้งที่อยู่คิดกันและแยกกันเรียงตัวอย่างเป็นระเบียบเมฆชนิดนี้อาจทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลดและปรากฏการณ์แถบสีหรือรุ้งประกอบด้วยผลึกน้ำแข็ง ฐานเมฆสูงเฉลี่ยประมาณ 7,000 เมตร')
         
     elif c_name == c_type[3]:
         st.title(f"**{c_name}** \n ")
         st.header('**เมฆเซอโรสเตรตัส หรือซีร์โรสเตรตัส (เมฆชั้นสูง):**')
         st.markdown('**เป็นเมฆชั้นสูง:** ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
         st.markdown('เมฆนี้ไม่ทำให้เกิดน้ำฟ้าบ่งบอกว่าลักษณะอากาศดีโดยเฉพาะหลังพายุฝนมีลักษณะเป็นก้อนเล็กๆเป็นหย่อมแผ่นหรือชั้นคล้ายเกล็ดก้อนกลมหรือม้วนมีทั้งสีเทาหรือทั้งสองสีบางครั้งอาจเห็นคล้ายปุยหรือฝ้าซึ่งประกอบด้วยละอองน้ำจำนวนมากมักทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ทรงกลดประกอบด้วยผลึกน้ำแข็ง เป็นเมฆที่เป็นระลอกคลื่นหรือก้อนกลมๆ เป็นแถวๆ ลักษณะคล้ายเกล็ดปลา เรียงกันเป็นระเบียบ ฐานเมฆสูงเฉลี่ยประมาณ 7,000 เมตร.')
         
     elif c_name == c_type[4]:
         st.title(f"**{c_name}** \n ")
         st.header('**เมฆเซอรัส หรือซีร์รัส (เมฆชั้นสูง):**')
         st.markdown('**เป็นเมฆชั้นสูง:** ในบริเวณเขตร้อนจะเกิดที่ความสูงระหว่าง 6-12 กิโลเมตร ส่วนใหญ่จะมีสีขาวหรือเทาอ่อน และเป็นเมฆซึ่งไม่ทำให้เกิดฝน ส่วนใหญ่จะเป็นผลึกน้ำแข็ง เพราะมีอุณหภูมิต่ำกว่าจุดเยือกแข็ง และมีความแปรปรวน.')
         st.markdown('หากเห็นเมฆประเภทนี้บ่งบอกได้เลยว่าวันนั้นลักษณะอากาศดี เมฆชนิดนี้อาจทำให้เกิดปรากฏการณ์วงแสงรอบดวงอาทิตย์หรือดวงจันทร์ แต่ยังไม่เต็มวงประกอบด้วยผลึกน้ำแข็ง มีลักษณะเป็นแผ่นบางสีขาวเจิดจ้า หรือสีเทาอ่อน มีรูปทรงอยู่หลายแบบ เช่น เป็นฝอยคล้ายขนนกบางๆ หรือขนสัตว์ หรือเป็นทางยาว ฐานเมฆสูงเฉลี่ย 10,000 เมตร.')
         
     elif c_name == c_type[5]:
         st.title(f"**{c_name}** \n **เมฆหางเครื่องบิน:**\n")
         st.markdown('**คอนเทรล:** มีสองชนิดทั้งสองชนิดล้วนเกิดจากการควบแน่นหรือการกลั่นตัวของไอน้ำในอากาศไปเป็นหยดน้ำเล็กๆในรูปแบบเดียวกับเมฆนั่นเอง คือ')
         st.markdown('**คอนเทรลที่เกิดจากไอพ่น (exhaustcontrails):** ไอร้อนที่ถูกพ่นออกมาปะทะกับอากาศที่อุณหภูมิต่ำและกลั่นตัวเป็นหยดน้ำในสุด เราจึงพบคอนเทรลได้บ่อยครั้ง ในขณะที่เครื่องบินกำลังบินอยู่บนระดับเพดานบินที่ค่อนข้างสูงเพราะว่าที่ระดับความสูงมากๆ อุณหภูมิของอากาศรอบๆจะลดต่ำลง')
         st.markdown('**คอนเทรลที่เกิดจากหลักอากาศพลศาสตร์ (aerodynamic contrails):** คอนเทรลชนิดนี้จะเกิดขึ้นที่ไหนก็ได้ที่อากาศมีความชื้นมากและความกดอากาศต่ำมากพอ แต่ที่ๆเราจะสามารถพบได้มากที่สุดก็คือบริเวณเหนือปีกเครื่องบิน (airfoils) ขณะที่เครื่องบินกำลังลงจอด (landing) เนื่องจากบริเวณเหนือปีกเครื่องบินจะมีความกดอากาศต่ำลงเป็นพิเศษ ซึ่งเป็นผลมาจากการออกแบบรูปทรง ของปีกและ มุมปะทะระหว่างลมและปีกเครื่องบิน (angle of attack) ในระหว่างการทำการบิน โดยคอนเทรลชนิดนี้จะเกิดขึ้นมาและหายไปทันทีที่อากาศกลับเข้าสู่อุณหภูมิปกติ (ambient temperature) ')
         st.image('./PContrail.jpg')

     elif c_name == c_type[6]:
         st.title(f"**{c_name}** \n")
         st.header('**เมฆคิวมูโลนิมบัส(เมฆก่อตัวในแนวตั้ง):**')
         st.markdown('**เมฆก่อตัวในแนวตั้ง:** เป็นเมฆที่ก่อตัวในแนวตั้งที่มีความรุนแรง และเกิดขึ้นโดยฉับพลัน ความสูงของฐานเมฆอยู่ที่ประมาณ 500 เมตร ส่วนยอดเมฆมีความสูงไม่แน่นอน บางครั้งอาจสูงถึงระดับเมฆชั้นสูง')
         st.markdown('มีลักษณะเป็นเมฆที่หนา มีขนาดใหญ่มากปกคลุมพื้นที่ครอบคลุมทั่วจังหวัด ก่อตัวสูงมาก บางครั้งยอดเมฆจะแผ่ออกคล้ายรูปทั่ง ทำให้เกิดฝนตกหนัก ลมแรง ฟ้าแลบ ฟ้าร้อง เกิดพายุฟ้าคะนอง บางครั้งมีลูกเห็บตก จึงมักเรียกว่า เมฆฟ้าคะนอง')

     elif c_name == c_type[7]:
         st.title(f"**{c_name}** \n")
         st.header('**เมฆคิวมูลัส(เมฆก่อตัวในแนวตั้ง):**')
         st.markdown('**เมฆก่อตัวในแนวตั้ง:** เป็นเมฆที่ก่อตัวในแนวตั้งที่มีความรุนแรง และเกิดขึ้นโดยฉับพลัน ความสูงของฐานเมฆอยู่ที่ประมาณ 500 เมตร ส่วนยอดเมฆมีความสูงไม่แน่นอน บางครั้งอาจสูงถึงระดับเมฆชั้นสูง')
         st.markdown('**เมฆชนิดนี้มี 2 ลักษณะ:**')
         st.markdown('หากอยู่เป็นก้อนเดี่ยวๆ ลักษณะปุกปุยก่อต่อในแนวตั้งสีขาว ส่วนฐานเมฆจะมีสีเทา มีความหนาที่สามารถบดบังแวงจากดวงอาทิตย์ได้ หากเห็นลักษณะนี้แสดงว่าอากาศดี ')
         st.markdown('หากประกอบกับท้องฟ้ามีสีฟ้าเข้ม เมื่อเริ่มจับตัวเป็นกลุ่มก้อนใหญ่ขึ้นมา จะทำให้เกิดฝนฟ้าคะนองขึ้นได้ ซึ่งอาจทำให้เกิดปรากฏการณ์ทรงกลดและรุ้งได้')

     elif c_name == c_type[8]:
         st.title(f"**{c_name}** \n")
         st.header('**นิมโบสเตรตัส(เมฆชั้นต่ำ):**')
         st.markdown('**เมฆชั้นต่ำ:** เกิดขึ้นที่ระดับต่ำกว่า 2 กิโลเมตร ประกอบด้วยอนุภาคน้ำเกือบทั้งหมด')
         st.markdown('เมฆแผ่นสีเทา เมื่อสังเกตเห็นเมฆชนิดนี้มักจะเกิดฝนพรำระยะเวลาราว 2-3 ชั่วโมง ฝนตกแต่แดดออกหรือทำให้เห็นสายฝนที่ตกลงมาจากฐานเมฆแต่ยังไม่มีพายุฝนฟ้าคะนอง ลักษณะจะเป็นเมฆสีเทาซึ่งจะทำให้ท้องฟ้าดูสลัวเนื่องจากมีความหนาที่ทำให้บังดวงอาทิตย์ได้')

     elif c_name == c_type[9]:
         st.title(f"**{c_name}** \n")
         st.header('**เมฆสเตรโตคิวมูลัส(เมฆชั้นต่ำ): **')
         st.markdown('**เมฆชั้นต่ำ:** เกิดขึ้นที่ระดับต่ำกว่า 2 กิโลเมตร ประกอบด้วยอนุภาคน้ำเกือบทั้งหมด')
         st.markdown('ทำให้เกิดฝนตกเล็กน้อย มักเกิดขึ้นในเวลาที่อากาศไม่ดี ลักษณะเป็นก้อนลอยติดกันเป็นแพ ไม่มีรูปทรงที่ชัดเจน มีช่องว่างระหว่างก้อนเล็กน้อย สีเทาหรือค่อนข้างขาว แต่ก็มีส่วนที่มืดครึ้มอยู่ด้วย และอาจทำให้เห็นปรากฏการณ์ดวงอาทิตย์หรือดวงจันทร์ทรงกลดได้เช่นกัน และปรากฏการณ์แถบสี หรือรุ้ง')

     elif c_name == c_type[10]:
         st.title(f"**{c_name}** \n ")
         st.header('**เมฆสเตรตัส หรือสตราตัส(เมฆชั้นต่ำ): **')
         st.markdown('**เมฆชั้นต่ำ:** เกิดขึ้นที่ระดับต่ำกว่า 2 กิโลเมตร ประกอบด้วยอนุภาคน้ำเกือบทั้งหมด')
         st.markdown('เมฆชนิดนี้มักเกิดขึ้นช่วงเช้า หรือหลังฝนตก และอาจทำให้เกิดฝนละอองขึ้นได้ ลักษณะเป็นแผ่นบาง สีเทา ลอยเหนือพื้นไม่มากนัก เช่น เมฆปกคลุมยอดเขา เป็นต้น')
         




else:
     st.error(f"Sorry, pls take another image") #result display
st.markdown(' ')
st.markdown('________________________________________________')
st.markdown(' ')
st.markdown('Link ต่างๆที่เกี่ยวข้อง')
st.markdown('Medium : https://medium.com/@Chinochi/cloud-type-classification-cloud-class-ระบบแยกประเภทของเมฆ-6c5233f1ab8')
st.markdown('GitHub : https://github.com/Mhapong/cloud_classify')
st.markdown('________________________________________________')
st.markdown('**ขอบคุณพี่ๆทีมAI Builders ที่ช่วยพัฒนาผมจากคนที่ทำอะไรไม่เป็นสักอย่างรู้สึกไม่มั่นใจกับตัวเองกลายมาเป็นเป็นคนที่สามารถทำชิ้นงานชิ้นแรกได้และถึงชิ้นงานนี้จะเป็นก้าวเล็กๆแต่มันอาจเป็นก้าวสำคัญในชีวิตของผม**')
st.markdown('________________________________________________')
st.markdown('ขอบคุณเว็บไซต์ที่ให้ความรู้')
st.markdown('ทรูปลูกปัญญา : https://www.trueplookpanya.com/learning/detail/33793')
st.markdown('กรุงเทพธุระกิจ : https://www.bangkokbiznews.com/lifestyle/892728')