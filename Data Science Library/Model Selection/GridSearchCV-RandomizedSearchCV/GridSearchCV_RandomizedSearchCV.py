""" GridsearcvCV

    Hiperparametre, parametrelerden farkli olarak hiperparametreler, modelin egitilmesi sirasinda
    ogrenilmez. Modelleme asamasindan oncesinde yetkili kisi tarafindan belirlenir. Ornegin KNN;
    KNN algoritmasi tahmin edilmek istenen degere en yakin k tane komsusuna bakarak siniflandirma
    yapar. Burada k sayisi ve metric(uzaklik metrigi) yetkili kisi tarafindan belirtilmesi gereken
    modelin performansini arttirabilecek hiperparametrelerdendir.
    
    GridSearchCV;
    - Modelde denenmesi istenen hiperparametreler ve degerleri icin butun kombinasyonlar ile ayri ayri
    model kurulur ve belirtilen metrige gore en basarili hiperparametreler belirlenir.
    - Kucuk veri setlerinde ve sadece birkac hiperparametre denenmek istendiginde cok iyi calisir.
    - Buyuk veri seti ile calisildiginda ya da denenecek olan hiperparametre sayisi ve degeri
    arttirildiginda kombinasyon sayisi da katlanarak artacaktir.
    
    RandomizedSearchCV;
    - Rastgele olarak bir hiperparametre seti secilir ve Cross Validation ile model kurularak
    test edilir. Belirlenen sure ya da iterasyon sayisi kadar bu adimlar devam eder.
    - Buyuk veri setlerinde daha az maliyetle GridSearchCV yontemiyle elde edilen en iyi skora
    yakin performans gosterecek hiperparametre setlerini belirleyebilir.
    - Daha genis bir hiperparametre alani tarayabilir.
    - Her ne kadar optimum hiperparametre setine yaklassa da tum olasi kombinasyonlari tek tek
    denemedigi icin en iyi performansi gosteren hiperparametre setini bulmayi garanti edemez.
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

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
csv = cross_val_score(svc, X=X_train, y=y_train, cv=4)
print('Modelin Basarim Ortalamasi: ', csv.mean())
print('Modelin Basarim Standart Sapmasi: ', csv.std())

# GridSearchCV - RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
parameters = [{'C':range(1, 11), 'kernel':['linear']},
              {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[1, 0.5, 0.1, 0.01, 0.001]}]


# Aldiklari Parametreler
# estimator : algoritmanin uretilen nesnesi
# param_grid/param_distributions : denenecek parametrelerin oldugu liste, tuple, dict
# scoring : Neye gore skorlanacak? orn; accuracy
# cv : Kac katlamali olacak?
# n_jobs : Ayni anda calisacak is sayisi.
gs = GridSearchCV(estimator=svc, # SVM
                  param_grid=parameters,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

rs = RandomizedSearchCV(estimator=svc, # SVM
                        param_distributions=parameters,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result_gs = grid_search.best_score_ # En iyi skor
best_param_gs = grid_search.best_params_ # En iyi parametreler
print('GridSearchCV Params: ', best_param_gs)
print('GridSearchCV Basarim: ', best_result_gs)
print()

random_search = rs.fit(X_train, y_train)
best_result_rs = random_search.best_score_ # En iyi skor
best_param_rs = random_search.best_params_ # En iyi parametreler
print('RandomizedSearchCV: ', best_param_rs)
print('RandomizedSearchCV Basarim: ', best_result_rs)
