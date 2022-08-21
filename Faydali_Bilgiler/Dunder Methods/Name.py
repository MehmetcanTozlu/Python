"""Dunder(Double Under) Methods

    Python calismaya basladiginda, kodlari okumadan once bazi ozel degiskenlerin ve niteliklerin atamasini gerceklestirir.
Bu niteliklerden birisi de __name__'dir. Python'da bulunan her modulun bir __name__ degerli ozelligi vardir.
__name__ ozelliginin degeri, programi dogrudan calistirdigimizda '__main__' degerine atanir. Aksi taktirde __name__ degeri modulun
ismini icerir. Yani __name__ iki farkli deger alabilir. Bunlar;
1. __main__ ozelligi
2. Calistirildigi modulun adi degerlerini alir.

if __name__ == '__main__'

__name__ ile kodumuza cift yonluluk kattigimiz soylenebilir. Kodumuz dogrudan calistiginda islevi farkli, modul olarak calistiginda
islevi farkli olur.
Kodumuz dogrudan calisiyor ise if icindeki satirimiz calismis oluyor. Aksi durumlarda yani modul ile calistiginda
if __name__ == '__main__' olmayacagi icin if blogu calismiyor.
"""

def topla(x, y):
    print(x + y)


def carp(x, y):
    print(x * y)


if __name__ == '__main__': # Bu dosyada calisirsa if satiri calisir
    topla(5, 10)

else: # Bu dosyayi modul olarak alan dosyalarda else blogu calisir
    carp(10, 20)
