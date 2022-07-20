""" OOP - Inheritance

Miras alinan sinif = Parent/Base/Super* Class
Miras alan sinif = Sub*/Child Class

super() = miras alinan sinifin ozelliklerini, metotlarini miras alan sinif tarafindan kullanilmasini saglar.
overriding = miras alinan sinifin metodunun aynisinin miras alan sinif icinde tanimlanmasi

__bases__ = bir snifin hangi siniftan miras aldigini gosterir

Coklu Kalitim; bir sinifin birden fazla sinifi miras almasi.
Python coklu kalitimi destekler. Ancak kullanmamiz cok tavsiye edilmez.

MRO(Method Resolution Order) = Birden fazla Class'tan miras alindiginda Miras alma sirasi(cozum sirasi)

Multi-Level Inheritance = Bir Class birden fazla Class'tan miras aliyorsa ve miras aldigi Class'lar
Parent Class'tan miras aliyorsa, birden fazla miras alan Class Parent Class'tan miras almasina gerek kalmaz.
"""


# Parent Class
class Kullanici():

    def __init__(self, ad, kAdi, parola):
        self.ad = ad
        self.kAdi = kAdi
        self.parola = parola

    def girisYap(self):
        print(self.kAdi + ' kullanicisi giris yapti.')


# Sub Class Moderator sinifi coklu kalitim ornegi icin uretilmistir
class Moderator(Kullanici):

    def __init__(self, ad, kAdi, parola):
        super().__init__(ad, kAdi, parola)

    def icerikKontrol(self):
        print('Icerik Kontrol Edildi.\nModerator...')
    
    def girisYap(self):
        print(self.kAdi + ' isimli moderator giris yapti.')


# Sub Class Abone sinifi coklu kalitim ornegi icin uretilmistir
class Abone(Kullanici):

    def __init__(self, ad, kAdi, parola):
        super().__init__(ad, kAdi, parola)

    def aboneOl(self):
        print(f'{self.kAdi} abone oldu.')
    
    def girisYap(self):
        print(self.kAdi + ' kullanici isimli aboneniz giris yapti.')


# Child Class
class Yonetici(Abone, Moderator, Kullanici):  # Kullanici sinifindan miras aldik
    # Coklu kalitimda miras alma sirasi onemlidir! Parent sinif en sona yazilir
    # Abone ve Moderator siniflari Kullanici Parent sinifindan miras aldigi icin onu yazmayabiliriz!

    def __init__(self, ad, kAdi, parola, mail):
        # super metodu miras alinan sinifin attributes erismemizi sagliyor
        super().__init__(ad, kAdi, parola)
        self.mail = mail

    # Overriding -> miras alinan sinifin metodunu miras alan sinif icinde tanimlamak
    def girisYap(self):
        super().girisYap()  # miras alinan sinifin metodunu cagiriyoruz
        print(f'{self.kAdi} yoneticisi sisteme giris yapti.')


kullanici = Kullanici('Mehmet', 'mehmet123', '123asd')
kullanici.girisYap()
print('**************************************')
yonetici = Yonetici('Ali', 'yilmaz', '123456', 'ali@mail.com')
yonetici.girisYap()
print('**************************************')
# Yonetici sinifinin kimden miras aldigini gosterir.
print(Yonetici.__bases__)
print(Kullanici.__bases__)
print('**************************************')
yonetici.icerikKontrol()
yonetici.aboneOl()
# miras alma isleminin oncelik sirasi Abone sinifinda old icin super().girisYap() Abone sinifindaki
# girisYap() metodunu cagirir!
yonetici.girisYap()
print('**************************************')
# Yonetici sinifinin miras almadaki cozum sirasini gosterir, oncelik sirasi ilkden sona
print('Yonetici Sinifinin Metot Cozum Sirasi(MRO): ', Yonetici.__mro__)
print('**************************************')
print('Yonetici Sinifinin Miras Aldigi Siniflar: ' + str(Yonetici.__bases__))

print('**************************************')
print(isinstance(kullanici, Kullanici)) # kullanici object'i Kullanici'nin instance mi? True-False doner
print(isinstance(kullanici, Yonetici))
