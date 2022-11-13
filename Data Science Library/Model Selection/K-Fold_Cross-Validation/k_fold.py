""" K-Fold Cross Validation

    K-Fold Cross Validation, siniflandirma modellerinin degerlendirilmesi ve modelin egitilmesi icin veri
    setini parcalara ayirma yontemlerinden biridir.
    Ornegin, elimizde 1000 kayitlik veri seti olsun. Bu veri setinin bir kismini egitim bir kismini da
    test olarak kullanmak isteyelim. Basitce, %75'i egitim %25'i test icin ayirmaktir.
    Ancak veri parcalanirken verinin dagilimina bagli olarak modelin egitim ve testinde bazi sapmalar(bias)
    ve hatalar olusabilir. K-Fold Cross Validation, veriyi belirlenen k sayisina gore esit parcalara boler.
    Her bir parcanin hem egitim hem de test icin kullanilmasini saglar. Boylelikle dagilim ve parcalanmalardan
    kaynaklanan sapma ve hatalar minimuma indirilir. Ancak modeli k kadar egitmek ve test etmek icin
    veri isleme yuk ve zaman ister. Bu durum kucuk veri setlerinde cok sorun olmasa da buyuk olcekli
    setlerinde maliyetli olabilir.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')

X = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM
from sklearn.svm import SVC
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Cross Validation
from sklearn.model_selection import cross_val_score
# estimator : kullanacagimiz algoritma nesnesi(burada svc).
# X : Hangi X degerinden hangi Y degerini tahmin edecek. Tahmin icin kullanilacak deger.
# y : Tahmin edilecek deger.
# cv : kac katlamali olacagi.
cvs = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=4)
print('Basarimin Ortalamasi: ', cvs.mean()) # Ne kadar yuksek o kadar iyi
print('Basarimin Standart Sapmasi: ', cvs.std()) # Ne kadar dusuk o kadar iyi
