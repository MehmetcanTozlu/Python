import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Pre Processing

df = pd.read_csv('Missing_Data_Ulke.csv') # Veri Yukleme

# Missing Data
from sklearn.impute import SimpleImputer # Gerekli Kutuphanemiz
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # nan olanlar icin bulunduklari sutunun ortalamasini yazdir
age = df.iloc[:, 1:4].values # nan olan degerleri degiskene atiyoruz
imputer = imputer.fit(age) # 10. satirdaki islemimizi age icin uygula
age = imputer.transform(df.iloc[:, 1:4]) # age degiskenindeki nan degerleri ortalama ile degistir

# Encoder(Categoric -> Numeric)
country = df.iloc[:, 0:1].values # Categoric olan sutunlari degiskene atadik
from sklearn import preprocessing # Gerekli Kutuphanemiz
le = preprocessing.LabelEncoder() # Categoric olan verilere ilk sayisal degerleri atiyoruz
country[:, 0] = le.fit_transform(df.iloc[:, 0:1])
ohe = preprocessing.OneHotEncoder() # Column'larda Label'leri tasir ve index'i olan Label'e 1 olmayana 0 yazdirir
country = ohe.fit_transform(country).toarray()

# Numpy Array'lerinin Pandas Data Frame'e donusumleri ve Frame'lerin Birlesip Veri Setinin son halini olusturma
countryFrame = pd.DataFrame(data=country, index=range(22), columns=['FR', 'TR', 'US'])
ageFrame = pd.DataFrame(data=age, index=range(22), columns=['Boy', 'Kilo', 'Yas'])
gender = df.iloc[:, 4:5].values
genderFrame = pd.DataFrame(data=gender, index=range(22), columns=['Cinsiyet'])

# Concat islemi ile Data Frame'leri Birlestirme
result = pd.concat([countryFrame, ageFrame], axis=1)
dataFrame = pd.concat([result, genderFrame], axis=1) # Veri Setimizin Son Hali

# Train ve Test Islemleri - Verilerin Olceklenmesi
from sklearn.model_selection import train_test_split
# Veri Setinin %33'u test olacak sekilde degiskenlere atama islemi
x_train, x_test, y_train, y_test = train_test_split(result, genderFrame, test_size=0.33, random_state=0)

# Verileri Algoritmamizin daha dogru calismasi icin uygun araliga cekiyoruz
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
