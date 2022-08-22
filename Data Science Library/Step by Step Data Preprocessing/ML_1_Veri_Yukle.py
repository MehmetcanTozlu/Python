import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ulke.csv')
print(df)
print(df.isnull().sum())

print('****************Eksik Veri Iceren Data Set**********************')

# df2 = pd.read_csv('Missing_Data_Ulke.csv')
# print(df2)

# # Yas sutununun ortalamasini NaN olanlara yazdiralim
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # yerine yazilacak degisken, yazilma prensibi
# yas = df2.iloc[:, 1:4].values # tum rowlari al, 1.'den 4'e kadar olan columnlar icin gecerli
# print(yas)

# imputer = imputer.fit(yas[:, 1:4]) # tum rowlar ve 1. column'dan 4. columna kadar degerleri al
# # fit fonksiyonu egitmek icin kullaniyoruz. Alicagi parametrede ogrenecek olan parametre
# # yas'in 1'den 4'e kadar olan column'larini ogrenmesini soyluyoruz
# yas[:, 1:4] = imputer.transform(yas[:, 1:4]) # nan degerleri SimpleImputer strategy icinde yazdigimiz mean'e gore yazdiriyoruz
# # fit ile ogrenip transform ile de ogrendigini uygulamasini soyluyoruz
# print(yas)
