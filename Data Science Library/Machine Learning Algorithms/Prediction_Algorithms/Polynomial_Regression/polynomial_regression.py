import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('maaslar.csv')

# Iliski kurmak istedigimiz degiskenler Egitim Seviyesi ve maas
# unvanlari Numeric olarak yaparsak Egitim verilerinin aynisi olacagi icin bunu yapmayacagiz

# Veri Setimizi Veri On Islemeye sokmuyoruz ve Verilerimizin hepsini train olarak verelim test'e veri ayirmayalim

x = df.iloc[:, 1:2]  # Egitim Seviyesi
y = df.iloc[:, 2:3]  # Maas
X = x.values
Y = y.values

# Ilk Linear Regression'da verilerimizi modelleyip nasil bir sonuc alicagiz bakalim
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='black')
plt.show()

# Polynomial Regressions
from sklearn.preprocessing import PolynomialFeatures
# 2. dereceden bir polinomsal ifadeyi temsil ediyor
poly_reg = PolynomialFeatures(degree=2)
# bagimsiz degiskenlerimizi polinomsal ifadelere transfor ediyoruz
x_poly = poly_reg.fit_transform(X)
# Formul -> B0*X^0 + B1*X^1 + B2*X^2 + .... + Bn*X^n
# x_poly -> Formulde; ilk Column -> B0'i  /  ikinci Column -> B1*X^1  /  ucuncu Column -> B2*X^2 'yi temsil ediyor
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y) # x^0, x^1, x^2'yi kullanarak y'yi ogren

plt.scatter(X, Y, color='r')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='k')
plt.show()

# Daha Yuksek Dereceli Bir Polinomial icin veri setimizi verelim
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
# Formul -> B0*X^0 + B1*X^1 + B2*X^2 + .... + Bn*X^n
# x_poly -> Formulde; ilk Column -> B0'i  /  ikinci Column -> B1*X^1  /  ucuncu Column -> B2*X^2 'yi temsil ediyor
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y) # x^0, x^1, x^2'yi kullanarak y'yi ogren

plt.scatter(X, Y, color='r')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='k')
plt.show()

# Predict - Linear Regression
print(lin_reg.predict([[11]])) # Egitim Seviyesi 11 ise
print(lin_reg.predict([[6.6]])) # Egitim Seviyesi 6.6 ise
print('*******************')
# Predict - Polinomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
