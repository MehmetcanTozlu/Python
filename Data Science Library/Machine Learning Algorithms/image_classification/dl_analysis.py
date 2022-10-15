import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

categories = ['Cat', 'Dog', 'Dinosor', 'Horse']

varsayilanData = []
hedefData = []

dataDir = 'C:/users/mehmet/desktop/Image_Classification_Example'

for i in categories:
    print(f'{i} Kategorisi Okunuyor...')
    path = os.path.join(dataDir, i) # Dosyadaki Klasorleri gezer
    # print(os.listdir(path)) # Tek Tek okunan resimleri gosterir
    for img in os.listdir(path):
        imgArray = imread(os.path.join(path, img))
        imgResize = resize(imgArray, (224, 224, 3))
        varsayilanData.append(imgResize.flatten())
        hedefData.append(categories.index(i))
    print(f'{i} Kategorisi Okundu.')

duzVeri = np.array(varsayilanData)
hedefVeri = np.array(hedefData)

df = pd.DataFrame(data=duzVeri)
df['Hedef'] = hedefVeri

independent = df.iloc[:, :-1]
dependent = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# Giris
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))

# Cikis
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(X_train,y_train, epochs=80)

loss = model.history.history['loss']
plt.plot(range(len(loss)), loss)

trainLoss = model.evaluate(X_train, y_train, verbose=0)
testLoss = model.evaluate(X_test, y_test, verbose=1)
print(trainLoss)
print(testLoss)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
