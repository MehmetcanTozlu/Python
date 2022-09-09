""" Upper Confidence Bound
Adim-1:
    Her turda(tur sayisi n olsun), he reklam/ilan alternatifi(i icin) asagidaki sayilar tutulur
        Ni(n): i sayili reklamin o ana kadar ki tiklanma sayisi
        Ri(n): o ana kadar ki i reklamindan gelen toplam odul
Adim-2:
    Yukaridaki bu iki sayidan, asagidaki degerler hesaplanir
        O ana kadar ki her reklamin/ilanin ortalama odulu -> Ri(n) / Ni(n)
        Guven araligi icin asagi ve yukari oynama potansiyeli -> di(n) sqrt( (3/2) * (log(n) / Ni(n)) )
Adim-3:
    En yuksek UCB degerine sahip olani aliriz.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ads_CTR_Optimisation.csv')

import math

N = 10000 # Veri Setindeki toplam veri sayisi
d = 10 # Column Sayisi
oduller = [0] * d # 10 elemanli liste, listenin her bir elemani 0. 10 ilanin odul degeri 0 / Ri(n)
tiklamalar = [0] * d # O ana kadar tiklamalar / Ni(n)
toplam = 0 # Toplam Odul
secilenler = []

for n in range(1, N):
    ad = 0 # Secilen Ilan
    max_ucb = 0 # Baslangic ucb degerimiz
    
    # max_ucb degerini bulan kod satirlari
    for i in range(0, d): # her bir satir icin hangi ilana tiklanacagini bulma
        """10 ilaninda teker teker degerlerine bak iclerinden en fazla UCB degerini bul."""
        
        if(tiklamalar[i] > 0): # max_ucb 0 olarak basladigi icin error vericek bu yuzden dongu 1 defa donsun
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt((3/2) * math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10

        if max_ucb < ucb: # max_ucb'den daha buyuk bir ucb cikarsa
            max_ucb = ucb
            ad = i
    
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = df.values[n, ad]
    oduller[ad] = oduller[ad] + odul
    toplam += odul
    
print('Toplam Odul: ', toplam)

plt.hist(secilenler)