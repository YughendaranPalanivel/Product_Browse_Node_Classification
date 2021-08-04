from sklearn.feature_extraction.text import CountVectorizer
import requests, zipfile, io
from keras import models
from keras import layers
from keras import metrics
import pandas as pd
import numpy as np
import csv
import nltk
nltk.download('stopwords')

#Downloading dataset
r = requests.get('https://s3-ap-southeast-1.amazonaws.com/he-public-data/dataset52a7b21.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(".//")

#Connecting to drive for saving ML models
from google.colab import drive
drive.mount('/content/content')

#Train data
df = pd.read_csv('//content//dataset//train.csv',usecols=['TITLE', 'BROWSE_NODE_ID'],escapechar = "\\", quoting = csv.QUOTE_NONE)
df = df.drop_duplicates(subset='TITLE', keep="last")
df = df.fillna("")
df['TITLE'] = df['TITLE'].apply(lambda s:s.lower() if type(s) == str else s)
df['TITLE'] = df['TITLE'].str.replace('\d+', '')
df['TITLE'] = df['TITLE'].str.replace('[^\w\s]','')
stop = nltk.corpus.stopwords.words('english')
df['TITLE'] = df['TITLE'].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
df['TITLE'] = df['TITLE'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1')
title = df['TITLE'].tolist()
id = df['BROWSE_NODE_ID'].tolist()
df = df.sort_values('BROWSE_NODE_ID')
Ymap = df.drop_duplicates(subset='BROWSE_NODE_ID', keep="last")
Ymap = Ymap['BROWSE_NODE_ID'].tolist()

#Test data
tdf = pd.read_csv('//content//dataset//test.csv',usecols=['PRODUCT_ID','TITLE'],escapechar = "\\", quoting = csv.QUOTE_NONE)
tdf = tdf.fillna("", inplace = True)
tdf['TITLE'] = tdf['TITLE'].apply(lambda s:s.lower() if type(s) == str else s)
tdf['TITLE'] = tdf['TITLE'].str.replace('\d+', '')
tdf['TITLE'] = tdf['TITLE'].str.replace('[^\w\s]','')
tdf['TITLE'] = tdf['TITLE'].apply(lambda words: ' '.join(word.lower() for word in str(words).split() if word not in stop))
tdf['TITLE'] = tdf['TITLE'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1')
title = tdf['TITLE'].tolist()
proid = tdf['PRODUCT_ID'].tolist()

#Saving cleaned data
df.to_csv('//content//dataset//train-2.csv', index=False)
tdf.to_csv('//content//dataset//test-2', index=False)

#Extracting keywords from train data

#Vectorization
cv = CountVectorizer()
vectorized = cv.fit(keywords)

#Building model
model = models.Sequential()
model.add(layers.Convolution1D(filters=8, kernel_size = 3, strides = 1, input_shape=(8693,1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Convolution1D(filters=4, kernel_size = 3, strides = 1))
model.add(layers.MaxPooling1D(2))
model.add(layers.Convolution1D(filters=2, kernel_size = 3, strides = 1))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(1000))
model.add(layers.Dense(9919, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',  
   optimizer = 'adam', metrics = ['accuracy'])

#Traing model
for epoch in range(0,5):
  print("__epoch__:",epoch)
  for i in range(0,2751000, 1000):
  x_train = []
  y_train = []
  for k in range(0,1000):
    x_train.append(np.array(vectorized.transform([title[i+k]]).toarray()[0]))
    y_train.append([0]*9919)
    y_train[k][Ymap.index(id[i+k])] = 1
    x_train = np.reshape(x_train,(1000,8693,1))
    y_train = np.array(y_train)
    his = model.fit(x_train, y_train,epochs=1, batch_size=25)
  x_train = []
  y_train = []
  i = 2751000
  for k in range(0,513):
    x_train.append(np.array(vectorized.transform([title[i+k]]).toarray()[0]))
    y_train.append([0]*9919)
    y_train[k][Ymap.index(id[i+k])] = 1
  x_train = np.reshape(x_train,(513,8693,1))
  y_train = np.array(y_train)
  his = model.fit(x_train, y_train,epochs=1, batch_size=25)
  model.save('//content//model//my_model.h5')

model.save('//content//model.h5')

#Prediction file
test_iteration_count = df.shape[0]
fileName = "//content//dataset//submission.csv"
headers = 'PRODUCT_ID,BROWSE_NODE_ID'
with open(fileName, 'w', encoding="utf-8") as file:
  file.write(headers + '\n')
  for i in range(0,110775):
    x_test = np.array(vectorized.transform([test_title[i]]).toarray()[0])
    x_test = np.reshape(x_test,(1,8693,1))
    pred = model.predict(x_test)
    ans = pred.argmax()
    file.write(f'{proid[i]},{Ymap[ans]}\n')
