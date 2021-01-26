import streamlit as st
from PIL import Image
from clf import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

#st.title("Cassava Leaf Disease Classification App")
#st.write("")

st.sidebar.markdown("""<h1  style="color:Green;">
             Home </h1>""", 
            unsafe_allow_html=True)

st.sidebar.subheader('Want to know your Cassava leaves are healthy or infected? ')

Mode = st.sidebar.selectbox('Choose', ('Home', 'Upload image file'))



st.sidebar.text('©r.hasina.angelina')

if Mode == 'Home':
    st.markdown("""<h1 align="center" style="color:Green;">
            🌄 Cassava Leaf Disease Classification 🌄</h1>""", 
            unsafe_allow_html=True)
    st.image('image/header.png', use_column_width=True)
    
    st.subheader('About')
    st.write("""
    Cassava Leaf disease classification 
    can help farmers to save their Cassava crops from diseases.
    This application was made to identify the disease type with a cassava plant using a photo from a relatively inexpensive camera. 
    And distinguish between several diseases that cause material harm 
    to the food supply of many African countries. 
    In some cases the main remedy is to burn the infected plants to prevent further spread, 
    so this project can help to make a rapid automated turnaround quite useful to the farmers.
    """)
    st.subheader('Overview of the studied Disease types')
    st.write("Let's have a look of how your Cassava leaf should look like when healthy or affected with a particular disease.")
    st.markdown("""<h3 style="color:Green;">
                ✳️ Healthy</h3>""", 
                unsafe_allow_html=True)
    st.image('image/Healthy.png', use_column_width=True)
    
    
    st.markdown("""<h3 style="color:Green;">
                ✳️ Cassava Brown Streak Disease (CBSD)</h3>""", 
                unsafe_allow_html=True)
    st.image('image/CBSD.png', use_column_width=True)
    st.markdown("""<h4 style="color:Brown;">
                🧪  Symptoms</h4>""", 
                unsafe_allow_html=True)
    
    st.write('Caused by virus, the symptoms on leaves is that it becomes chlorotic or necrotic vein banding in mature leaves which may merge later to form large yellow patches.')
    
    
    
    st.markdown("""<h3 style="color:Green;">
                ✳️ Cassava Mosaic Disease (CMD)</h3>""", 
                unsafe_allow_html=True)
    st.image('image/CMD.png', use_column_width=True)
    st.markdown("""<h4 style="color:Brown;">
                🧪  Symptoms</h4>""", 
                unsafe_allow_html=True)
    
    st.write('Caused by virus, discolored pale green, yellow or white mottled leaves which may be distorted with a reduced size; in highly susceptible cassava cultivars plant growth may be stunted, resulting in poor root yield and low quality stem cuttings. Note that infected plants can express a range of symptoms and the exact symptoms depend on the species of virus and the strain as well as the environmental conditions and and the sensitivity of the cassava host.')
    
    
    st.markdown("""<h3 style="color:Green;">
                ✳️ Cassava Bacterial Blight (CBB)</h3>""", 
                unsafe_allow_html=True)
    st.image('image/CBB.png', use_column_width=True)
    st.markdown("""<h4 style="color:Brown;">
                🧪  Symptoms</h4>""", 
                unsafe_allow_html=True)
    
    st.write('Caused by Bacterium, small, angular, brown, water-soaked lesions between leaf veins on lower surfaces of leaves; leaf blades turning brown as lesion expands; lesions may have a yello halo; lesions coalesce to form large necrotic patches; defoliation occurs with leaf petioles remaining in horizontal position as leaves drop; dieback of shoots; brown gum may be present on stems, leaves and petioles.')
    
    
    st.markdown("""<h3 style="color:Green;">
                ✳️ Cassava Green Mottle (CGM)</h3>""", 
                unsafe_allow_html=True)
    st.image('image/CGM.png', use_column_width=True)
    st.markdown("""<h4 style="color:Brown;">
                🧪  Symptoms</h4>""", 
                unsafe_allow_html=True)
    
    st.write('Caused by Arachnid, yellow stipping of leaves; chlorotic spots on leaves; chlorosis of entire leaves; if infestation is very high then leaves may be stunted and deformed; terminal leaves may die and drop from plant; pest responsible is a tiny green mite.')




# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
MODEL_DIR = './ResNext model/cassava-models-res/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
#TRAIN_PATH = './cassava-leaf-disease-classification/train_images'
#TEST_PATH = './image'


# # CFG

# In[2]:


# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    num_workers=8
    model_name='resnext50_32x4d'
    size=512
    batch_size=32
    seed=2020
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    inference=True


# # Library


# In[7]:


# ====================================================
# Library
# ====================================================
import sys
sys.path.append('./pytorch-image-models-master')

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Utils

# In[8]:


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'inference.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

#LOGGER = init_logger()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


# # Data Loading

# In[9]:


#test = pd.read_csv('./sample_submission.csv')
#test.head()


# # Dataset

# In[10]:


# ====================================================
# Dataset
# ====================================================

class TestDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = cv2.imread(self.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


# # Transforms

# In[11]:


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    if data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# # MODEL

# In[12]:


# ====================================================
# MODEL
# ====================================================
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# # Helper functions

# In[21]:


# ====================================================
# Helper functions
# ====================================================
def load_state(model_path):
    model = CustomResNext(CFG.model_name, pretrained=False)
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'], strict=True)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model']
    except:  # multi GPU model_file
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


# # inference

# In[22]:


# ====================================================
# inference
# ====================================================
model = CustomResNext(CFG.model_name, pretrained=False)
states = [load_state(MODEL_DIR+f'{CFG.model_name}_fold{fold}.pth') for fold in CFG.trn_fold]
#test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
#test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
#                         num_workers=CFG.num_workers, pin_memory=True)
#predictions = inference(model, states, test_loader, device)
## submission
#test['label'] = predictions.argmax(1)
##test[['image_id', 'label']].to_csv(OUTPUT_DIR+'submission.csv', index=False)
##test.head()
#st.subheader('Test image prediction')
#st.write("""
#This is the prediction for the submission of this model
#""")
#st.write(test)

if Mode == 'Upload image file':
    st.markdown("""<h1 align="center" style="color:Green;">
            🌄 Cassava Leaf Disease Classification 🌄</h1>""", 
            unsafe_allow_html=True)
    file_up = st.file_uploader("Upload the image of your cassava leaf !", type=['png', 'jpg', 'jpeg'])

    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
      
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        transform=get_transforms(data='valid')
        if transform:
            augmented = transform(image=img)
            img = augmented['image']
        img = DataLoader(img, batch_size=CFG.batch_size, shuffle=False, 
                             num_workers=CFG.num_workers, pin_memory=True)
    #    st.write(len(img))
        predictions = inference(model, states, img, device)
        st.subheader('Prediction Probability')
        st.write(predictions)
        with open('cassava_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
    
    #    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    #    _, indices = torch.sort(out, descending=True)
    #    [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
       
        if np.argmax(predictions) == 4:
    #        st.write("Your Cassava leaf is healthy")
            st.markdown("""<h3 align="center" style="color:Green;">ℹ️  Result:   Your Cassava leaf is <b><u>Healthy</u></b>.</h3>""", unsafe_allow_html=True)
    
        else:
    #        st.write("Your Cassava leaf is sick with the type of disease : ",  #classes[np.argmax(predictions)])
            st.markdown(f"""<h3 align="center" style="color:Red;">🛑  Result:   Your Cassava leaf is <b><u>Infected</u></b> with the type of disease : <b>{classes[np.argmax(predictions)]}</b></h3>""", unsafe_allow_html=True)     
        
    
        
st.sidebar.markdown("""
                    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <center><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
    .fa {
      width: 15px;
      height: 15px;
      display: inline-block;
      padding: 20px;
      font-size: 20px;
      cursor: pointer;
      text-align: center;
      text-decoration: none;  
      border: none;
      border-radius: 50%
    }

    </style>
    </head>
    <body><center>

    <!--<h2>Style Social Media Buttons</h2>-->

    <!-- Add font awesome icons -->


    <!-- <h2>Image Links</h2>

    <p>The image is a link. You can click on it.</p> -->


    <!--
    <a href="https://wa.me/+261342531720" target="_blank">
      <img src="https://cdn0.iconfinder.com/data/icons/tuts/256/whatsapp.png" 
      alt="Whatsapp" style="width:25px;height:25px;border:0">
    </a> -->

    <a href="https://web.facebook.com/hana.angelina.58/" target="_blank">
      <img src="https://cdn4.iconfinder.com/data/icons/free-social-media-icons/512/Facebook.png" 
      alt="Facebook" style="width:22px;height:22px;border:0">
    </a>

    <a href="mailto:angelina@dsi-program.com" target="_blank">
      <img src="https://cdn2.iconfinder.com/data/icons/once-again/48/Gmail.png" 
      alt="Mail" style="width:25px;height:25px;border:0">
    </a>

    <a href="https://github.com/AngelHa" target="_blank">
      <img src="https://cdn4.iconfinder.com/data/icons/miu-hexagon-shadow-social/60/github-hexagon-shadow-social-media-256.png" 
      alt="Whatsapp" style="width:25px;height:25px;border:0">
    </a>

    <!-- <p>We have added "border:0" to prevent IE9 (and earlier) from displaying a border around the image.</p> </p> -->


     </center>     
    </body>
    </html>

                    """, unsafe_allow_html=True
                    )
    
