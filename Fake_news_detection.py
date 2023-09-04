
# Fake News Detection using RNN and LSTM Networks.

# Importing the required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import re
import gensim


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,GRU,SimpleRNN
from tensorflow.keras.layers import Conv1D,MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


# Function to read the data


def read_data(data):
    Data_read=pd.read_csv(data)
    return Data_read

# Analyzing the real news data


Real_data=read_data('True.csv')


Real_data.head()

Real_data.columns
sns.countplot(x='subject',data=Real_data)


# Visualizing the most occured words in the real data

real_text=''.join(Real_data['text'].tolist())
wordcloud=WordCloud().generate(real_text)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# Analyzing the fake news data


Fake_data=read_data('Fake.csv')
Fake_data.head()

Fake_data.columns

sns.countplot(x='subject',data=Fake_data)

# Visualizing the most occured words in the fake data

fake_text=''.join(Fake_data['text'].tolist())
wordcloud=WordCloud().generate(fake_text)
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# Preprocessing the data (Cleaning the data-Removing Reuters info)

# For real data
anonymous_data=[]
for i, row in enumerate(Real_data.text.values):
  try:
    known=row.split('-',maxsplit=1)
    known[1]
    # Checking if data came from twitter (120 characters max )
    assert(len(known[0])<120)
  except:
    anonymous_data.append(i)

print(len(anonymous_data))


Real_data.iloc[anonymous_data]

# Dropping the missing values

for i in range(len(Real_data)):
  if (Real_data.text[i]==' '):
    print(i)

Real_data=Real_data.drop(8970, axis=0)

publisher=[]
text_data=[]

for i, row in enumerate(Real_data.text.values):
  if i in anonymous_data:
    text_data.append(row)
    publisher.append('anonymous')
  else:
    known=row.split('-',maxsplit=1)
    publisher.append(known[0].strip())
    text_data.append(known[1].strip())
    
Real_data['publisher']=publisher
Real_data['text']=text_data


Real_data.sample()

# Removing the null data values in Fake News Dataset

empty_indices=[i for i,text in enumerate(Fake_data.text.tolist()) if 
                str(text).strip()=='']
                

Fake_data.iloc[empty_indices].head()


# Text is missing because, it is in the title column itself, so merging both columns in both real and Fake dataset


Real_data['text']=Real_data['title']+" "+ Real_data['text']
Fake_data['text']=Fake_data['title']+" "+ Fake_data['text']

# Converting all text to lower case

Real_data['text']=Real_data['text'].apply(lambda i: str(i).lower())
Fake_data['text']=Fake_data['text'].apply(lambda i: str(i).lower())

# Giving class labels to both Fake and Real Dataset

Real_data['class']=1
Fake_data['class']=0

# Extracting only the required columns


Real_data=Real_data[['text','class']]
Fake_data=Fake_data[['text','class']]


# Combining both the datasets to pass it into the Neural Network for training

Appended_data=Real_data.append(Fake_data,ignore_index=True)


# Removing special characters

Appended_data['text']=Appended_data['text'].str.replace('[^\w\s]', '')

Appended_data.head()

# Performing the Word to Vector (Word2Vec) Vectorization i.e, embedding on the text data to pass it through the network for training.

labels=Appended_data['class'].values
Input=[i.split() for i in Appended_data['text'].tolist()]

vector_model_class=gensim.models.Word2Vec(sentences=Input, size=80, 
                                          window=8,min_count=1)

# Tokenizing the data


# Instantiating the tokenizer class
tokenizer=Tokenizer()
tokenizer.fit_on_texts(Input)
Input=tokenizer.texts_to_sequences(Input)

# Visualizing the histogram of the data to perform padding


plt.hist([len(i) for i in Input], bins=1000)
plt.show()

# Pad the text length of sequences to 800 (If length greater than 800, make it to 800, if less than 800, add zeroes(padding)) as there are very few sequences with length more than 800


Input=pad_sequences(Input, maxlen=800)

# Initializing weight matrix with the data itself, and performing training on that


weight_init=np.zeros(((len(tokenizer.word_index))+1,80))
for word, index in tokenizer.word_index.items():
  weight_init[index]=vector_model_class.wv[word]


# Creating the Neural Network Model (RNN)

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Training on RNN (Recurrent Neural Network)

# Creating the Sequential RNN Model

model_1=Sequential()
model_1.add(Embedding(len(tokenizer.word_index)+1,output_dim=80,
            weights=[weight_init],input_length=800,trainable=True))
model_1.add(SimpleRNN(units=100,activation='tanh',dropout=0.24))
model_1.add(Dense(1,activation='sigmoid'))
model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model_1.summary()


# Training the model


# Splitting the training and testing data
Input_train, Input_test,Label_train, Label_test=train_test_split(Input,labels)

model_1.fit(Input_train, Label_train, validation_split=0.3, epochs=4)

# Checking the classification report on Test set

pred_labels=model_1.predict((Input_test)>=0.5).astype(int)

print("Accuracy score is {0}".format(accuracy_score(Label_test, pred_labels)))

# ## Training on LSTM (Long Short Term Memory Neural Network)

model_2=Sequential()
model_2.add(Embedding(len(tokenizer.word_index)+1,output_dim=80,
            weights=[weight_init],input_length=800,trainable=True))
model_2.add(LSTM(units=150,dropout=0.24))
model_2.add(Dense(1,activation='sigmoid'))
model_2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model_2.summary()
model_2.fit(Input_train, Label_train, validation_split=0.2, epochs=5)
pred_labels_2=model_2.predict((Input_test)>=0.5).astype(int)

print("Test set accuracy on LSTM Model is {0}".
      format(accuracy_score(Label_test, pred_labels_2)))



