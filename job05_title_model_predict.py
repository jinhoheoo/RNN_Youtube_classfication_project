import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import datetime



df = pd.read_csv('./data/data_exam_20240129.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/label_encoder.pickle','rb') as f:
    label_encoder = pickle.load(f)

label = label_encoder.classes_

print(label)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('./models/Youtube_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 29:
        tokened_x[i] = tokened_x[i][:29]
print(tokened_x)

x_pad = pad_sequences(tokened_x, 29)

model = load_model('./models/Youtube_category_classification_model_0.931034505367279.h5')
preds = model.predict(x_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts

print(df)

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'

print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))
filtered_df = df[df['OX'] == 'X']
print(filtered_df)
filtered_df.to_csv('./data_filtered.csv', index=False)
