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

learn = 'Cloud_resnet50_fastai.pkl'
learn.save()