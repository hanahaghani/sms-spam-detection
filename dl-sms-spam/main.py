#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader


data=pd.read_csv("data/processed-spam.csv",encoding="latin-1")
data.info()

x=data['v2']
y=data['v1']

#data encoder & TF-ID
vectorized=TfidfVectorizer()
x=vectorized.fit_transform(x)
x=pd.DataFrame(x.toarray())

encoder=LabelEncoder()
y=encoder.fit_transform(y)

print(type(y))
print(type(x))

#split the data

x_train_full,x_test,y_train_full,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

x_train,x_valid,y_train,y_valid=train_test_split(
    x_train_full,y_train_full,random_state=42,test_size=0.2
)

#converd pd to numpy after to torch
dataset=[
    x_test,x_train,x_valid
]
datasets=[
    y_test,y_train,y_valid
]

dataset=[data.to_numpy() for data in dataset]

datasets=[torch.tensor(data,dtype=torch.float32) for data in datasets]
dataset=[torch.tensor(data,dtype=torch.float32) for data in dataset]

x_test,x_train,x_valid=dataset
y_test,y_train,y_valid= datasets

y_train=y_train.unsqueeze(1)
y_test=y_test.unsqueeze(1)
y_valid=y_valid.unsqueeze(1)

print(y_train.shape)

#tensor dataset

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)
valid_dataset=TensorDataset(x_valid,y_valid)

#dataloader

train_dataloader=DataLoader(
    train_dataset,batch_size=32,shuffle=True
)
test_dataloader=DataLoader(
    test_dataset,batch_size=32
)
valid_dataloader=DataLoader(
    valid_dataset,batch_size=32
)

#model
torch.manual_seed(42)
n_epoch=30
criteion=nn.MSELoss()
