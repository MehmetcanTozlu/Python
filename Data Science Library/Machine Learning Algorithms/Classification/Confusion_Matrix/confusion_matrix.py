import pandas as pd

df = pd.read_csv('Ulke.csv')
df2 = df.loc[5:, :] # ilk 5 veriyi cikarttik

x = df2.iloc[:, 1:4].values
y = df2.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# Kutuphanemizi import ediyoruz
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # gercek degerler,  tahmin edilen degerler
print(cm)

"""
Veri Setimizin tamamini aldigimizda confusion matrix hic dogru sonuc elde edemedi
Veri Setimizden ilk 5 veriyi cikartip modele verdigimizde ise confusion matrix hepsini dogru bildi
"""
