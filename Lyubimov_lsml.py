from __future__ import print_function, division

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from tensorboardX import SummaryWriter

import os
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")


# Обработка данных вдохновлена:
# https://www.kaggle.com/ateplyuk/inat2019-starter-keras-efficientnet

ann_file = '/data/inaturalist/train2019.json'
with open(ann_file) as data_file:
        train_anns = json.load(data_file)
        
valid_ann_file = '/data/inaturalist/train2019.json'
with open(valid_ann_file) as data_file:
        valid_anns = json.load(data_file)

train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})

valid_anns_df = pd.DataFrame(valid_anns['annotations'])[['image_id','category_id']]
valid_img_df = pd.DataFrame(valid_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})

df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
df_valid_file_cat = pd.merge(valid_img_df, valid_anns_df, on='image_id')

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, transform=True):

        self.data_frame = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['file_name'])
        image = cv2.imread(img_name)
        label = self.data_frame.iloc[idx]['category_id']
        
        data_transformation = transforms.Compose([
                                                     transforms.ToPILImage(),
                                                     transforms.Resize(size=(28, 28)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                          std=[0.229, 0.224, 0.225])
                                                 ])
        if self.transform:
            image = data_transformation(image)
            
        return image, label

def train_model(
        model, optimizer, criterion, scheduler, train_dataset, val_dataset, log_writer, batch_size=32, epochs=10):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=15)
    
    train_step = 0
    val_step = 0
    
    for epoch in range(epochs):
        
        scheduler.step()
        model.train()
        
        for x, y in tqdm(train_loader):
            
            x = x.cuda()
            y = y.cuda()
            
            optimizer.zero_grad()
            output = model(x)
            pred = torch.max(output, 1)[1]

            loss = criterion(output, y)            
            loss.backward()
            optimizer.step()
            
            acc = torch.eq(pred, y).float().mean()
            
            log_writer.add_scalar('train/CrossEntropyLoss', loss.item(), train_step)
            
            train_step += 1

        model.eval()
                
        for x, y in tqdm(val_loader):

            x = x.cuda()
            y = y.cuda()
            
            with torch.no_grad():

                output = model(x)
                val_pred = torch.max(output, 1)[1]
                val_loss = criterion(output, y)
                val_acc = torch.eq(val_pred, y).float().mean()
            
            log_writer.add_scalar('validation/CrossEntropyLoss', val_loss.item(), val_step)

            val_step += 1

train_dataset = CustomDataset(df=df_train_file_cat, root_dir='/data/inaturalist/')
val_dataset = CustomDataset(df=df_valid_file_cat, root_dir='/data/inaturalist/')

#возьмем предобученную сеть
model_ft = models.resnet18(pretrained=True)

#переведем все параметры в режим False
for params in model_ft.parameters():
     params.requires_grad = False 
        
model_ft.fc = nn.Linear(in_features=512, out_features=1010, bias=True)
params_to_train = model_ft.fc.parameters()
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(params_to_train, lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

log_writer = SummaryWriter(os.path.join('/data/inaturalist/', 'all_logs'))

train_model(model_ft, 
            optimizer_ft, 
            criterion, 
            exp_lr_scheduler, 
            train_dataset, 
            val_dataset, 
            log_writer, 
            batch_size=320,
            epochs=12)