import pandas as pd

data=pd.read_csv('/home/pc/Documents/machine learning/sms-spam-detection/dl-sms-spam/data/spam.csv',encoding="latin-1")

data.info()

data=data.drop(columns=['Unnamed: 4','Unnamed: 3','Unnamed: 2'])

data.info()

data.to_csv("/home/pc/Documents/machine learning/sms-spam-detection/dl-sms-spam/data/processed-spam.csv",index=False)