import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.api import OLS

# 10 yil tecrubeli 100 puan almis CEO'yu modellerimize tahmin ettirelim

df = pd.read_csv('maaslar_yeni.csv')
# unvan satirini data frame'den cikaralim

print(df.corr())

x = df.iloc[:, 2:5]
y = df.iloc[:, -1]
X = x.values
Y = y.values

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
linear_pred = lin_reg.predict(X)

print('--------------------------------Linear Regression p-value---------------------')
model = OLS(linear_pred, X)
print(model.fit().summary())

# r2-score
print('Linear Regression r2-score: ', r2_score(Y, linear_pred))

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

print('--------------------------------Polynomial Regression p-value---------------------')
model2 = OLS(poly_pred, X)
print(model.fit().summary())

print('Polynomial Regression r2-score: ', r2_score(Y, poly_pred))

# Support Vector Regression
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
X_scaler = scaler1.fit_transform(X)
scaler2 = StandardScaler()
Y_scaler = scaler2.fit_transform(Y.reshape(-1, 1))

from sklearn.svm import SVR
sv_reg = SVR(kernel='rbf')
sv_reg.fit(X_scaler, Y_scaler)
sv_pred = sv_reg.predict(X_scaler)

print('--------------------------------SV Regression p-value---------------------')
model3 = OLS(sv_pred, X_scaler)
print(model3.fit().summary())

print('Support Vector Regression r2-score: ', r2_score(Y_scaler, sv_pred))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)
dt_pred = dt_reg.predict(X)

print('--------------------------------Decison Tree p-value---------------------')
model4 = OLS(dt_pred, X)
print(model4.fit().summary())

print('Decision Tree r2-score: ', r2_score(Y, dt_pred))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y)
rf_pred = rf_reg.predict(X)

print('--------------------------------Random Forest p-value---------------------')
model5 = OLS(rf_pred, X)
print(model5.fit().summary())

print('Random Forest r2-score: ', r2_score(Y, rf_pred))

print('--------------------------------------------------------------')

print('CEO Maasi Linear: ', lin_reg.predict([[10, 10, 100]]))
print('CEO Maasi Polynomial ', lin_reg2.predict(poly_reg.fit_transform([[10, 10, 100]])))
print('CEO Maasi SVR: ', sv_reg.predict([[10, 10, 100]]))
print('CEO Maasi Decision Tree: ', dt_reg.predict([[10, 10, 100]]))
print('CEO Maasi Random Forest: ', rf_reg.predict([[10, 10, 100]]))
