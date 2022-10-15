import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.metrics import mean_squared_error, f1_score, precision_score
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

Categories = ['Cat', 'Dog', 'Horse', 'Dinosor']

duzVeriler = []
hedefVeriler = []

dataDir = 'C:/Users/mehmet/Desktop/Image_Classification_Example'

for i in Categories:
    print(f'{i} Kategorisi Yukleniyor...')
    path = os.path.join(dataDir, i)
    #print(os.listdir(path))
    for img in os.listdir(path):
        imgArray = imread(os.path.join(path, img))
        imgResize = resize(imgArray, (150, 150, 3))
        duzVeriler.append(imgResize.flatten())
        hedefVeriler.append(Categories.index(i))
    print(f'{i} Kategorisi Basariyla Yuklendi...')

duz_veri = np.array(duzVeriler)
hedef_veri = np.array(hedefVeriler)

df = pd.DataFrame(duz_veri)
df['Hedef Veri'] = hedef_veri
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('Egitim ve Test verileri basariyla ayrildi...')

from sklearn.tree import DecisionTreeClassifier
log_reg = DecisionTreeClassifier()
log_reg.fit(X_train, y_train)
print('Model Basariyla Egitildi')
y_pred = log_reg.predict(X_test)
print('Model Basariyla Test Edildi')

print('Tahmin Edilen Veri: ')
print(y_pred)

print('Gercek Veri: ')
print(np.array(y_test))

print(f'Modelin Accuracy Orani: {accuracy_score(y_test, y_pred) * 100}')

url=input('Resmin URL\'si :')
img = imread(url)
plt.imshow(img)
plt.show()
img_resize=resize(img,(150,150,3))
l = [img_resize.flatten()]
olasilik = log_reg.predict_proba(l)

for ind, val in enumerate(Categories):
    print(f'{val} = {olasilik[0][ind]*100}%')
print("Tahmin Edilen Resim : " + Categories[log_reg.predict(l)[0]])
