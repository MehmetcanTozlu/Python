import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri Setimizi On Islemeden Gecirelim
df = pd.read_csv('Churn_Modelling.csv')

print(df.corr())
print(df.describe())
print(df.isnull().sum())

X = df.iloc[:, 3:13].values # independent variables(11)
y = df.iloc[:, -1].values # dependent variable(1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([('ohe', OneHotEncoder(dtype=float), [1])],
                        remainder='passthrough')
X = ohe.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Artificial Neural Network
import keras # gerekli kutuphanemisi ekliyoruz
from keras.models import Sequential # keras'a bir ANN olustur diyoruz
from keras.layers import Dense # ANN'de ki katmanlarimizi ekliyoruz(neurons)

# Alt satirda ANN'imizi olusturduk
classifier = Sequential()

# Modelimizi olusturmak icin de Sequential'dan urettigimiz nesnemize eklemeler yaparak
# modelimizi olusturuyoruz

# units : Pozitif tamsayi, cikti uzayinin boyutlulugu, gizli katmandaki noron sayisi(kisiden kisiye degisebilir).
# activation : Kullanilacak aktivasyon fonksiyonu. Hicbir sey belirtmezsek, etkinleştirme uygulanmaz(a(x)=x).
# use_bias : Boolean, katmanin bir sapma vektoru kullanip kullanmadigi.
# kernel_initializer : kernel_agirlik matrisi icin baslatici.
# bias_initializer : Bias vektoru icin baslatici.
# kernel_regularizer : Agirlik matrisine uygulanan duzenleyici islevi.
# bias_regularizer : Bias vektorune uygulanan duzenleyici islevi.
# Activity_regularizer : Katmanin ciktisina uygulanan duzenleyici islevi.
# kernel_constraintkernel : Agirlik matrisine uygulanan kisitlama islevi.
# bias_constraint : Bias vektorune uygulanan kisitlama islevi.

# input_dim : Giris katmanimizda ki veri sayisi(independent variables).

# Input
classifier.add(Dense(units=6, activation='relu', input_dim=11)) # units = X + y / 2 Giris Katmanimiz
classifier.add(Dense(units=6, activation='relu'))

# Output
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile
    # optimizer : Ogrenme oranini kontrol eder. Model icin optimal agirliklarin ne kadar hizli hesaplandigi belirler.\
        # Daha kucuk oran daha iyi ogrenme saglar ancak zaman ister.\
        # Sinapsisler uzerindeki degerlerin nasil optimize edilecegini belirtir.(Adam, RMSprop, SGD, Adadelta Adamax, Nadam etc.)
    # loss : Modelin hata oranini ayni zamanda basarimini olcen fonksiyondur. Loss fonk. temelde modelin yaptigi\
        # tahminin, gercek degerden(ground truth) ne kadar farkli oldugunu hesaplamaktir.
        # Iyi bir modelden beklentimiz; 0'a yakin loss degerinin olmasidir. Neden 0 değil? Regularization.\
        # (binary_crossentropy, categorical_crossentropy, crossentropy, mean_squared_error, mean_absolute_error etc.)
    # metrics : Model tarafindan degerlendirilecek metriklerin listesi.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
classifier.fit(x=X_train, y=y_train, epochs=50)

# Testing
y_pred = classifier.predict(X_test) # Oransal olarak deger uretir

y_pred = (y_pred > 0.5) # y_pred > 0.5 => T, y_pred < 0.5 => F

from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, classification_report
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

m_a_e = mean_absolute_error(y_test, y_pred)
print(m_a_e)

print(classification_report(y_test, y_pred))
