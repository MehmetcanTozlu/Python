import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maaslar.csv')

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:3]

X = x.values
Y = y.values

# Kutuphanemiz
from sklearn.ensemble import RandomForestRegressor
# n_estimators = Cizilecek Decision Tree Miktari
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X), color='black')

print(df, '\n')
print('6.6 Egitim Icin ->  ', rf_reg.predict([[6.6]]))
