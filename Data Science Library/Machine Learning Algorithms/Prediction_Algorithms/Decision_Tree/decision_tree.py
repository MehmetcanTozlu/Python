import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maaslar.csv')

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:3]

X = x.values
Y = y.values

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, r_dt.predict(X), color='black')
plt.show()

print(r_dt.predict([[11]])) # Egitim Seviyesi 11'i tahmin ettiyoruz
print(r_dt.predict([[6.6]])) # Egitim Seviyesi 6.6'i tahmin ettiyoruz
print(r_dt.predict([[7]])) # Egitim Seviyesi 6.6 olanla ayni maasi aliyor yani bu 2 deger ayni grupta yer aliyor demektir
