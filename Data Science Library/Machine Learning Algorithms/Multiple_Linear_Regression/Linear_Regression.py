import pandas as pd
import numpy as np

df = pd.read_csv('Ulke.csv')

ulke = df.iloc[:, 0:1].values
# Categoric Degerleri Numeric Yapalim
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(df.iloc[:, 0:1]).toarray()

frame1 = pd.DataFrame(data=ulke, index=range(22), columns=['FR', 'TR', 'USA'])
frame2 = df[['boy', 'kilo', 'yas']]
cinsiyet = df.iloc[:, 4:5].values

def convert(gender):
    if gender == 'e':
        return 0
    else:
        return 1

cins = list(map(convert, cinsiyet))
cins = np.array(cins)
cins = pd.DataFrame(data=cins, index=range(22), columns=['Cinsiyet'])

frame3 = pd.concat([frame1, frame2], axis=1)
dataFrame = pd.concat([frame3, cins], axis=1)

# train ve test olarak veri setimizi split edelim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(frame3, cins, test_size=0.33, random_state=0)

# Modelimizi olusturalim
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) # x_test'ten  y_test'i tahmin etmesini ogreniyoruz

# Yeni bir ornek girelim(DataFrame'de ilk eleman)
newValue = np.array([0,1,0,130,30,10]).reshape(-1, 6)
predict2 = regressor.predict(newValue)

# Hata oranina bakalim
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_test, y_pred)
print(error)

# Cinsiyet sutunu icin bir modelleme yaptik simdi de boy icin yeni bir model olusturalim
# boy sutununu veri setimizin son hali olan dataFrame'den cekelim

boy = dataFrame[['boy']]
dataFrame2 = dataFrame.drop(['boy'], axis=1) # boy sutununu dataFrame'den sildik
# boy -> Bagimli Degisken, dataFrame2 -> Bagimsiz Degisken

x_train, x_test, y_train, y_test = train_test_split(dataFrame2, boy, test_size=0.33, random_state=0)

# Modelimiz
regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_pred2 = regressor2.predict(x_test)

errorYas = mean_absolute_error(y_test, y_pred2)

# Variable Selection - Bacwark Elimination - p-Value
# Featur'larin Basarimlarini Olcelim;
import statsmodels.api as  sm

# tamami 1'lerden olusan 22 satir 1 sutun'luk bir matrix olustur ve bunu axis=1 olacak sekilde dataFrame2'ye yerlestir
X = np.append(arr=np.ones((22,1)).astype(int), values=dataFrame2, axis=1)
# her bir Column'u ifade edecek bir liste olusturalim ve bu liste uzerinden eleme yaparak ilerleyelim
X_list = dataFrame2.iloc[:, [0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(boy, X_list).fit() # istatiksel degerlerimizi cikaralim
print(model.summary())

# gorundugu gibi x5 yani 5. sutunumuz olan yas sutunu en yuksek p-degerine sahiptir. Bu yuzden ilk silecegimiz deger odur.
print('\n------------------------------Yas Sutunu Silindi-----------------------------\n')
X_list2 = dataFrame2.iloc[:, [0,1,2,3,5]].values
X_list2 = np.array(X_list2, dtype=float)
model2 = sm.OLS(boy, X_list2).fit()
print(model2.summary())
# 0.031 degeriyle en yuksek cinsiyet sutunu cikti 0.05'den kucuk old. icin istersek kalabilir ama biz silelim

print('\n----------------------------Yas-Cinsiyet Silindi-------------------------\n')
X_list3 = dataFrame2.iloc[:, [0,1,2,3]].values
X_list3 = np.array(X_list3, dtype=float)
model3 = sm.OLS(boy, X_list3).fit()
print(model3.summary())
