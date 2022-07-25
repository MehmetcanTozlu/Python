"""Polymorphism -> Cok Bicimlilik

Polymorphism; bir niteligin birden fazla kullanim seklinin olmasidir.

Fonksiyonlarda Polymorphism; ornegin len fonksiyonu;
len('Merhaba')
len([1, 2, 3, 4])
len((1, 2, 3))
len({'veri-1':'Kitap', 'veri-2':'Kalem'})
goruldugu gibi len fonksiyonun birden fazla kullanim sekli vardir. Bu Polymorphism'e ornektir.

Operatorlerde Polymorphism; + operatorunun;
metinsel ifadeler arasinda birlestirme islemi yaparken sayisal ifadelerde toplama yapar.
'Ankara' + 'Baskent' => Ankara Baskent
3 + 5 => 8

Classlarda Polymorphism;
class icindeki methodlarimizin isimleri ayni oldugunda, ayni isimde ancak farkli islevlere sahip oldugundan
polymorphism'e ornektir.

Inheritanceda Polymorphism;
Kalitim alan siniflar, Super Class'daki methodlari override ettigi taktirde Polymorphism yapmis olurlar.
"""

# Class'larda ki Polymorphism ornegi
from ast import Mod


print('Class\'larda ki Polymorphism ornegi\n')


class Kopek:
    def __init__(self, ad):
        self.ad = ad

    def bilgi(self):
        print('Merhaba benim adim: {0}'.format(self.ad))

    def ses(self):
        print('Hav Hav')


class Kedi:
    def __init__(self, ad):
        self.ad = ad

    def bilgi(self):
        print('Merhaba benim adim {0}'.format(self.ad))

    def ses(self):
        print('Miyav')


class Kus:
    def __init__(self, ad):
        self.ad = ad

    def bilgi(self):
        print('Merhaba benim adim {0}'.format(self.ad))

    def ses(self):
        print('Cik Cik')


kopek = Kopek('Kopek')
kedi = Kedi('Kedi')
kus = Kus('Kus')

for i in (kopek, kedi, kus):
    i.bilgi()
    i.ses()
# Class'larda ki Polymorphism ornegi
print('*********************************************')


print('Inheritance\'da ki Polymorphism ornegi\n')
# Inheritance'da Polymorphism ornegi


class Kullanici(object):
    def __init__(self, ad, kAdi, parola):
        self.ad = ad
        self.kAdi = kAdi
        self.parola = parola

    def girisYap(self):
        print(f'{self.kAdi} kullanicisi sisteme giris yapti.')


class Moderator(Kullanici):
    def __init__(self, ad, kAdi, parola):
        super().__init__(ad, kAdi, parola)

    def girisYap(self):  # override islemi
        print(f'{self.kAdi} Moderator sisteme giris yapti.')


class Gozetmen(Kullanici):
    def __init__(self, ad, kAdi, parola):
        super().__init__(ad, kAdi, parola)

    def girisYap(self):  # override islemi
        print(f'{self.kAdi} Gozetmen sisteme giris yapti.')


v1 = Moderator('Ali', 'Ali123', '123456')
v2 = Gozetmen('Veli', 'Veli456', '456789')

liste = [v1, v2]
for i in liste:
    i.girisYap()
# Inheritance'da Polymorphism ornegi
