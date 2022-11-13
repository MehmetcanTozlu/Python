""" Boyut Indirgeme / PCA - Principal Component Analysis(Birincil Bilesen Analizi) 

    Veri Biliminde boyut indirgeme, bir verinin yuksek boyutlu bir uzaydan,
    dusuk boyutlu bir uzaya, anlamini kaybetmeyecek sekilde donusturulmesidir.
    Yuksek boyutlu bir veriyi islemek daha fazla islem yuku gerektirir.
    Bu yuzden, yuksek sayida gozlemin ve degiskenin incelendigi
    sinyal isleme, konusma tanima, noroinformatik, biyoinformatik, gurultu filtreleme,
    gorsellestirme, oznitelik cikarimi, oznitelik eleme/donusturme, borsa analizi
    gibi alanlarda boyut indirgeme sikca kullanilir.
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Wine.csv')

X = df.iloc[:, 0:-1].values # independent variables
y = df.iloc[:, -1].values # dependent variables

# Train ve Test kumelerine bolduk
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling Islemi
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # kac kolona indirmek istediigimizi girdi olarak verdik

X_train_pca = pca.fit_transform(X_train) # train kumemizi yeniden boyutlandirma islemi
X_test_pca = pca.transform(X_test) # test kumemizi yeniden boyutlandirma islemi

# Modelimizi Olusturalim
from sklearn.linear_model import LogisticRegression

# PCA'dan once
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# PCA'dan sonra
log_reg_pca = LogisticRegression(random_state=0)
log_reg_pca.fit(X_train_pca, y_train)
y_pred_pca = log_reg_pca.predict(X_test_pca)

from sklearn.metrics import accuracy_score, confusion_matrix
print('PCA\'dan once Accuracy: ', accuracy_score(y_test, y_pred))
print('PCA\'dan once Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

print('\nPCA\'dan sonra Accuracy: ', accuracy_score(y_test, y_pred_pca))
print('PCA\'dan sonra Confusion Matrix:\n', confusion_matrix(y_test, y_pred_pca))
