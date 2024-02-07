import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./Youtube_titles_20240129.csv')
#print(df.head())
#df.info()

X = df['titles']
Y = df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y) #라벨 부여
print(labeled_y[:])
label = label_encoder.classes_  #부여된 라벨 정보를 확인
print(label)
with open('./models/label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f) #파이썬 데이터형으로 저장 리스트 -> 리스트로 불러온다.
onehot_y = to_categorical(labeled_y)
print(onehot_y[:])
#print(X[:5])
okt = Okt()
#for i in range(len(X[:5])):
 #   X[i] = okt.morphs(X[i])  #그냥 짜르기만 한다.
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #원형으로 바꾸어 주고
    if i % 1000:
        print(i)
#print(X[:5]) #한글자는 학습이 불가능, 접속사, 감탄사 는 의미 없음


stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = [] # j는 문장
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
#print(X[:5]) #불필요한 문자들을 삭제하고 리스트에 저장

#각 단어에 번로를 부여 (같은 형태소에 같은 번호를 붙여 주었다.)
token = Tokenizer()
token.fit_on_texts(X) #라벨을 붙이기
tokened_x = token.texts_to_sequences(X) #문장을 숫자 번호로 만듬
wordsize = len(token.word_index) + 1 #index가 1붙어 만들어 진다. +1를 더한 이유는 0을 쓰기 위함
#print(tokened_x)
print(wordsize) #사실 모든 총 숫자는 33개

with open('./models/Youtube_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0 #최대값을 찾는 코드
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max) #앞에 다가 0을 채워 길이를 맞추기 위함 (뒤로 가는게 학습에 좋기 때문에)
print(x_pad)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size = 0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
xy = np.array(xy, dtype=object)
np.save('./Youtube_data_max_{}_wordsize_{}'.format(max, wordsize), xy) #저장