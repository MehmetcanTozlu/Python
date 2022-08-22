import numpy as np
import pandas as pd

df = pd.read_csv('Missing_Data_Ulke.csv')

from sklearn.impute import  SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
missingDF = df.iloc[:, 1:4].values

imputer = imputer.fit(missingDF)
missingDF = imputer.transform(missingDF)

print('*************Categorical to Numerical****************')
ulkeCol = df.iloc[:, 0:1].values
print(ulkeCol)

# kutuphanemizi ekleyelim
from sklearn import preprocessing
# LabelEncoder -> categorical ifadeyi numeric yapar
le = preprocessing.LabelEncoder() # Nesnesini olusturuk
ulkeCol[:, 0] = le.fit_transform(df.iloc[:, 0])
print(ulkeCol)

# OneHotEncoder -> Numeric olan degerleri bulunduklari yerleri 1 digerlerini 0 yapan bir binary vektor yapar
ohe = preprocessing.OneHotEncoder() # Nesnesini olusturuduk
ulkeColBinaryVector = ohe.fit_transform(ulkeCol).toarray()
print(ulkeColBinaryVector)
