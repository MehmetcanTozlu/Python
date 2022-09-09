"""Thompson Sampling

Adim-1:
    Her bir aksiyon icin asagidaki 2 sayiyi hesaplamaliyiz.
        Ni1(n): o ana kadar odul olarak 1 gelmesi sayisi
        Ni0(n): o ana kadar odul olarak 0 gelmesi sayisi
Adim-2:
    Her ilan icin asagida verilen Beta dagiliminda bir rastgele sayi uretiyoruz.
        Teta = T
	Beta = B
	
	Ti(n) = B(Ni^1(n) + 1, Ni^0(n) + 1)
Adim-3:
    En yuksek Beta degerine sahip olani aliriz.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

df = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
toplam = 0
secilenler = []
birler = [0] * d # Ni1
sifirlar = [0] * d # Ni0

for n in range(1, N):
    max_th = 0
    ad = 0
    
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    
    secilenler.append(ad)
    odul = df.values[n, ad]
    
    if odul == 1:
        birler[ad] += 1
    else:
        sifirlar[ad] += 1
    
    toplam = toplam + odul

print('Toplam Odul: ', toplam)
plt.hist(secilenler)
plt.show()
