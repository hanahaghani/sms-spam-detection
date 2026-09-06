#importing library
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader

#import utils
from utils.model_preprocessing import spamclassifier_linear_relu
from utils.visualization import plot_training_history,visualize_new_data

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

#model 1
torch.manual_seed(42)

model1=spamclassifier_linear_relu(x_train.shape[1])
n_epoch=20
criterion=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(
    model1.parameters(),lr=0.001,weight_decay=0.01
)
scheduler=torch.optim.lr_scheduler.StepLR(
    optimizer,step_size=5,gamma=0.5
)
#training loop
train_losses=[]
valid_losses=[]
print('-----spamclassifier_linear_relu-----')
best_valid_loss=float('inf')
patience=3
counter=1

#________train dataset_________
for epoch in range(n_epoch):
    model1.train()
    total_train_loss=0
    for x_batch,y_batch in train_dataloader:
        optimizer.zero_grad()
        prediction=model1(x_batch)
        loss=criterion(prediction,y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_loss=total_train_loss/len(train_dataloader)
    train_losses.append(train_loss)
#_______validation_________
    model1.eval()
    total_valid_loss=0
    with torch.no_grad():
        for x_batch,y_batch in valid_dataloader:
            prediction=model1(x_batch)
            loss=criterion(prediction,y_batch)
            total_valid_loss+=loss.item()
        valid_loss=total_valid_loss/len(valid_dataloader)
        valid_losses.append(valid_loss)

    scheduler.step()

    if valid_loss<best_valid_loss:
        best_valid_loss=valid_loss
        counter=0
    else:
        counter+=1

    if counter>=patience:
        print("early stop!!!!!!!!!!!!!!")
        break
#------print----
    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {train_loss:.4f}, "
        f"Validation Loss = {valid_loss:.4f}"
    )
#_______test____
model1.eval()
total_test_loss=0

with torch.no_grad():
    for x_batch,y_batch in test_dataloader:
        prediction=model1(x_batch)
        loss=criterion(prediction,y_batch)
        total_test_loss +=loss.item()
    
    avg_test_loss=total_test_loss/len(test_dataloader)
    print(f'test loss:{avg_test_loss:.4f}')

#plots
plot_training_history(train_losses,valid_losses,"img/train-valid-loss-linear.png")

#new data for test
new_sms = "hi,mylove how is your feeling?"
visualize_new_data(
    text=new_sms,
    model=model1,
    tfidf_vectorizer=vectorized,
    save_path="img/neural_network_path1.png"
)

new_sms1 = "Congratulations! You won a free prize!"
visualize_new_data(
    text=new_sms1,
    model=model1,
    tfidf_vectorizer=vectorized,
    save_path="img/neural_network_path.png"
)

#save model
torch.save(model1.state_dict(),"model/sms_spam_linear_model.pth")