import numpy as np
import pandas as pd

df = pd.read_csv('Missing_Data_Ulke.csv')


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
missingData = df.iloc[:, 1:4].values
imputer = imputer.fit(missingData)
missingData = imputer.transform(missingData)

ulke = df.iloc[:, 0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(df.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()
ulkeBinaryVector = ohe.fit_transform(ulke).toarray()

print(ulkeBinaryVector)
print(missingData)
print(df.iloc[:, 4:5])

print('***************Verileri Birlestirme****************')
sonuc = pd.DataFrame(data=ulkeBinaryVector, index=range(22), columns=['FR', 'TR', 'USA'])
sonuc2 = pd.DataFrame(data=missingData, index=range(22), columns=['Boy', 'Kilo', 'Yas'])

cinsiyet = df.iloc[:, -1].values
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['Cinsiyet'])

# concat ile 3 farkli dataframe'i birlestirelim
s = pd.concat([sonuc, sonuc2], axis=1) # axis columnlarda birlestir demek
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)
