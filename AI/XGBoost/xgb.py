""" XGBoost - eXtreme Gradient Boosting

    Gradient Boosting(zayif ogrenicileri(weak learner) guzlu ogrenicilere(strong learner) donusturme yontemidir)
    algoritmasinin cesitli duzenlemelerle optimize edilmis yuksek performansli halidir.
    Daha az kaynak kullanarak ustun sonuclar elde etmek icin yazilim ve donanim optimizasyon teknikleri
    uygulanmistir. Karar Agaci tabanli algoritmalarin en iyisi olarak gosterilir.
    3 Onemli Ozelligi:
        1. Yuksek verilerde iyi performans gosterir(Calisma Performansi)
        2. Hizli Calisma
        3. *** Problem ve modelin yorumunun mumkun olmasi. ***
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')

X = df.iloc[:, 3:13].values # independent variables
y = df.iloc[:, 13].values # dependent variables

# Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = LabelEncoder()
X[:, 2] = le2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([('ohe', OneHotEncoder(dtype=float), [1])],
                        remainder='passthrough')
X = ohe.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Calisma dosyanizin ismini xgboost yapmayin! :)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
