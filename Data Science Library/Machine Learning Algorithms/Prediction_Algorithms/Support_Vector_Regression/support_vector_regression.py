import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maaslar.csv')

x = df.iloc[:, 1:2] # independent variables
y = df.iloc[:, 2:] # dependent variables
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='black')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='black')
plt.show()

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
X_scaler = scaler1.fit_transform(X)

scaler2 = StandardScaler()
Y_scaler = scaler2.fit_transform(Y)

# SVR icin kutuphanemizi ekleyelim
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf') # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
svr_reg.fit(X_scaler, Y_scaler)

plt.scatter(X_scaler, Y_scaler, color='red')
plt.plot(X_scaler, svr_reg.predict(X_scaler))
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
