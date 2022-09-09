"""
Ads_CTR_Optimisation.csv icindeki verileri analiz edelim.
En cok tiklanan reklam ilanini bulalim.
Random Selection; Rastgele secim yapar. Herhangi bir zeki secim yapmaz.
Kullanicinin tiklayabilecegi reklami dogru secip ona gosterebilirsek odul kazaniyoruz.
Aksi halde odulu kazanamiyoruz.
Veri Setimize bakacak olursak her bir column'un toplam tiklamasi birbirinden farkli.
Ad 1 = 1703 tiklanma, Ad 6 = 126 tiklanma vb. UCB bu tarz durumlarda avantajli oluyor.
Random Selection'da bu farkliligi yakalamamiz yok.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ads_CTR_Optimisation.csv')

# kutuphanemiz
import random

# 10k veri old icin 10k kadar donucek
N = 10000
d = 10 # Column sayisi rastgele bir sayi uret
toplam = 0
secilenler = []

for n in range(0, N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = df.values[n, ad] # veri kumesindeki n. satir 1 ise odul = 1
    toplam += odul

print('Odul Sayisi: ', toplam)
"""
Bu Algoritma rastgele urettigi verilerle tiklanan reklamlari tutturmaya yonelik calisiyor.
Toplam Odul Sayisida 1100 ile 1400 arasi degisiyor.
Bu Odul Miktarini UCB(Upper Confidence Bound) ile arttirmaya calisabiliriz.
"""
plt.hist(secilenler)
plt.show()
