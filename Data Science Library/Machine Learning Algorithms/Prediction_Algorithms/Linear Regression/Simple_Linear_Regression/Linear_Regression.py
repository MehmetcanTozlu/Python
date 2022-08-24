import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('aylara_gore_satis.csv')

# Veri Setimizde Categoric ve NaN degerler olmadigi icin direk train ve test asamasina gecebiliriz.

aylar = df[['Aylar']] # Direk DataFrame olarak df'den ayirdik
print(aylar)

satislar = df[['Satislar']]
print(satislar)

# train ve test olarak verilerimizi bolelim
from sklearn.model_selection import train_test_split
# bagimsiz degisken, bagimli degisken, teste ayrilacak oran
x_train,x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

# Olceklendirme Islemini Devre Disi Birakalim ve Cikan Sonuclar Daha Okunabilir Olucak Mi Kontrol Edelim;
# # Verileri Olceklendirelim
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)

# Modelimizi Olusturalim
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) # X_train'den Y_traini tahmin edicek

tahmin = lr.predict(x_test) # X_test'den Y_test'i dogru bulacilecekmi bakalim

# Hata Oranimizi Hesaplayalim
from sklearn.metrics import mean_absolute_error
predictArray = lr.predict(x_test)
errorRate = mean_absolute_error(y_test, predictArray) # dogru degeri ve dogru degerin tahminlerini verdik
print('Hata Orani: ', errorRate)


# Hesapladigimiz degerleri gorsellestirelim
# Cizdirecegimi degerleri sirayalim ki daha guzel bir gorsel elde edelim
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)

plt.title('Aylara Gore Satis')
plt.xlabel('Aylar')
plt.ylabel('Satis Miktari')


"""Islem Adimlarimiz;
X_train'den Y_train'i ogrendi. Modeli insa ederken bu 2 veriyi kullandi.
X_test'den de Y_test'i tahmin etti.
"""