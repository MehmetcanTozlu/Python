""" Boyut Indirgeme / LDA - Linear Discriminant Analysis(Dogrusal Ayirma Analizi) 

    Ozniteliklerin bir dogrusal birlesimini bularak veriyi siniflara ayirmaya
    yarayan yontem.
    LDA, bir verideki degiskenlerin, veriyi en iyi aciklayan dogrusal
    birlesimini incelemeleri acisindan temel bilesen analizi ve faktor analizi
    ile yakindan iliskilidir.
    LDA; verilen siniflari ayiran bir birlesim bulurken,
    Temel Bilesen Analizi; siniflari goz ardi eder.
    Faktor Analizi; sinif ici benzerlik yerine varyansi incelemesi ve gizli
    degiskenleri modellemesi le LDA'dan farklidir.
    LDA;
    - PCA benzeri bir boyut donusturme/indirgeme algoritmasidir.
    - PCA'den farkli olarak siniflar arasindaki ayrimi onemser ve maksimize
    etmeye calisir.
    - PCA bu acidan Unsupervised, LDA ise Supervised ozelliktedir.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Wine.csv')

X = df.iloc[:, 0:-1].values # independent variables
y = df.iloc[:, -1].values # dependent variables

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
# y_train'i vermemizdeki amac siniflarin oldugu boyutu almasi.
X_train_lda = lda.fit_transform(X_train, y_train) # PCA'de sadece X_train vermistik
X_test_lda = lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

log_reg_lda = LogisticRegression(random_state=0)
log_reg_lda.fit(X_train_lda, y_train)
y_pred_lda = log_reg_lda.predict(X_test_lda)

from sklearn.metrics import accuracy_score, confusion_matrix
print('LDA\'siz Accuracy: ', accuracy_score(y_pred, y_test))
print('LDA\'siz Confusion Matrix:\n', confusion_matrix(y_pred, y_test))


print('LDA\'li Accuracy: ', accuracy_score(y_pred_lda, y_test))
print('LDA\'li Confusion Matrix:\n', confusion_matrix(y_pred_lda, y_test))
