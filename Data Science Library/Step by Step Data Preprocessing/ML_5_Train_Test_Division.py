import numpy as np
import pandas as pd

df = pd.read_csv('Missing_Data_Ulke.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
missingValues = df.iloc[:, 1:4]
imputer = imputer.fit(missingValues)
missingValues = imputer.fit_transform(missingValues)

ulke = df.iloc[:, 0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(df.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

sonuc1 = pd.DataFrame(data=ulke, index=range(22), columns=['FR', 'TR', 'USA'])
sonuc2 = pd.DataFrame(data=missingValues, index=range(22), columns=['Boy', 'Kilo', 'Yas'])
cinsiyet = df.iloc[:, 4:5].values
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['Cinsiyet'])

s = pd.concat([sonuc1, sonuc2], axis=1)

ss = pd.concat([s, sonuc3], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
