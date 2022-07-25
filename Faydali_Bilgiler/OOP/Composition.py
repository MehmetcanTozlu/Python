"""Composition -> Birlesim

Has-A Relation
Is-part-of Relation

Iki sinif arasinda bir birinin parcasi olma anlami/bagi vardir(has-a).
Composition'da bu bag oldukca kuvvetlidir. Bir sinif yok old. diger sinifta yok olur.

Ornegin; bir universite ve fakulteleri dusunebiliriz. Universiteler fakultelere sahiptir.
Ancak universiteler yok old. fakultelerde yok olur.

Saglikli bir composition Constructor Icinde Yapilir!

Composition'un Inheritance'a karsi en buyuk avantaji daha esnek olmasi.

"""


class Fakulte:
    # 1-n iliskiyi bu sayede saglamis oluyoruz
    fakulteAdi = []
    
    # 1-1 iliskiyi bu sekilde sagliyoruz
    # fakulteAdi = None

    def fakulteOlustur(self, fakulteAdi) -> None:
        self.fakulteAdi.append(fakulteAdi) # 1-n iliski
        # self.fakulteAdi = fakulteAdi # 1-1 iliski

    def __repr__(self) -> str:
        return 'Fakulte Adi: ' + str(self.fakulteAdi)


class Universite:
    def __init__(self, uniAdi, yil) -> None:
        self.uniAdi = uniAdi
        self.yil = yil

        # Composition islemi / Uni yok old. fakultede yok olur
        # fakulte nesnesini kullanarak Fakulte sinifindaki ozelliklere erisebiliriz
        self.fakulte = Fakulte()

    def __repr__(self) -> str:
        return 'Universite Adi: ' + self.uniAdi + '\n' + str(self.fakulte)


uni1 = Universite('Firat Universitesi', '1975')
uni1.fakulte.fakulteOlustur('Muhendislik Fakultesi')
uni1.fakulte.fakulteOlustur('Tip Fakultesi')
# sadece tip fakultesini yazdirdi cunku 1-1 iliski vardir fakulte = None
# eger fakulte sinifinda ki fakulteyi bir dizi yaparsak o zaman 1-n iliski yapmis oluruz fakulte = []
print(uni1)

# Universite nesnesini yok ettigimizde fakulteye ulasamiyoruz(is-part-of)
# del(uni1)
