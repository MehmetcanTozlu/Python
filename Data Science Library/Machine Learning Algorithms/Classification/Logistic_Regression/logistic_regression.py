import pandas as pd

df = pd.read_csv('Ulke.csv')
# Boy, Kilo ve Yas'tan cinsiyet tahmin ettirmeye calisan modeli olusturalim

x = df.iloc[:, 1:4].values
y = df.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit -> egitme, transform -> ogrendigi egitimi uygulama
X_train = scaler.fit_transform(x_train) # x_train'den ogren ve transform et 
X_test = scaler.transform(x_test) # x_test icin yeniden ogrenme

# Kutuphanemizi import ediyoruz
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
