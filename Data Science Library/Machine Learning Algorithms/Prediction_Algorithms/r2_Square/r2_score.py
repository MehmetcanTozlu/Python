import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maaslar.csv')

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:3]

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x.values, y.values)

plt.scatter(x.values, y.values)
plt.plot(x.values, lin_reg.predict(x.values))
plt.show()

print('Linear Regression')
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[11]]))

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x.values)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y.values)

plt.scatter(x.values, y.values)
plt.plot(x.values, lin_reg2.predict(poly_reg.fit_transform(x.values)))
plt.show()

print('Polynomial Regression')
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

# Support Vector Regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
y_scaler = scaler.fit_transform(y)

from sklearn.svm import SVR
sv_reg = SVR(kernel='rbf')
sv_reg.fit(x_scaler, y_scaler)

plt.scatter(x_scaler, y_scaler)
plt.plot(x_scaler, sv_reg.predict(x_scaler))
plt.show()

print('Support Vector Regression')
print(sv_reg.predict([[6.6]]))
print(sv_reg.predict([[11]]))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x.values, y.values)

plt.scatter(x.values, y.values)
plt.plot(x.values, dt_reg.predict(x.values))
plt.show()

print('Decision Tree')
print(dt_reg.predict([[6.6]]))
print(dt_reg.predict([[11]]))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x.values, y.values)

plt.scatter(x.values, y.values)
plt.plot(x.values, rf_reg.predict(x.values))
plt.show()

print('Random Forest')
print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[11]]))


# r2 Score
from sklearn.metrics import r2_score
print('Random Forest r2-score Degeri')
print(r2_score(y.values, rf_reg.predict(x.values)))

print('Decision Tree r2-score Degeri')
print(r2_score(y.values, dt_reg.predict(x.values)))

print('Support Vector Regression r2-score Degeri')
print(r2_score(y_scaler, sv_reg.predict(x_scaler)))

print('Polynomial Regression r2-score Degeri')
print(r2_score(y.values, lin_reg2.predict(poly_reg.fit_transform(x.values))))

print('Linear Regression r2-score Degeri')
print(r2_score(y.values, lin_reg.predict(x.values)))
