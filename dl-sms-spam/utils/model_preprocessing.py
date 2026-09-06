#import library
import torch.nn as nn
import torch
import torch.nn.functional as f

#buliding models
#linear Relu
class spamclassifier_linear_relu(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.layer1=nn.Linear(input_size,256)
        self.layer2=nn.Linear(256,128)
        self.layer3=nn.Linear(128,64)
        self.output=nn.Linear(64,1)

        self.dropout=nn.Dropout(0.5)

    def forward(self,x):
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.dropout(x)

        x=self.layer2(x)
        x=torch.relu(x)
        x=self.dropout(x)

        x=self.layer3(x)
        x=torch.relu(x)
        x=self.dropout(x)

        x=self.output(x)
        
        return x

#linear leaky Relu
class spamclassifier_linear_leakyRelu(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.layer1=nn.Linear(input_size,256)
        self.layer2=nn.Linear(256,128)
        self.layer3=nn.Linear(128,64)
        self.output=nn.Linear(64,1)

    def forward(self,x):
        x=self.layer1(x)
        x=f.leaky_relu(x)

        x=self.layer2(x)
        x=f.leaky_relu(x)

        x=self.layer3(x)
        x=f.leaky_relu(x)

        x=self.output(x)

        return x

#batch norm
class spamclassifier_linear_batchnorm(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.layer1=nn.Linear(input_size,256)
        self.bn1=nn.BatchNorm1d(256)
        self.layer2=nn.Linear(256,128)
        self.bn2=nn.BatchNorm1d(128)
        self.layer3=nn.Linear(128,64)
        self.bn3=nn.BatchNorm1d(64)
        self.output=nn.Linear(64,1)

    def forward(self,x):
        x=self.layer1(x)
        x=self.bn1(x)
        x=f.relu(x)

        x=self.layer2(x)
        x=self.bn2(x)
        x=f.relu(x)

        x=self.layer3(x)
        x=self.bn3(x)
        x=f.relu(x)

        x=self.output(x)
        
        return x
