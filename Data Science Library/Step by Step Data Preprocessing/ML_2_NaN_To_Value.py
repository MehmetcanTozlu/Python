import numpy as np
import pandas as pd

df = pd.read_csv('Missing_Data_Ulke.csv')
print(df)
# kutuphanemizi ekleyelim
from sklearn.impute import SimpleImputer
# NaN olan verileri sec ve ortalamalarini yazdir komutunu degiskene atadik
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# sayisal degerleri degiskene atadik
missingData = df.iloc[:, 1:4].values

# fit parametresini egitim icin kullandik ve icine missingData arrayini verdik
imputer = imputer.fit(missingData)
# yukarida tanimladigimiz imputer degiskenini missingData icin uyguluyoruz
missingData = imputer.transform(missingData)   # bu satirda NaN olanlara sutunun otalamasini yazdirdik.
print(missingData)
