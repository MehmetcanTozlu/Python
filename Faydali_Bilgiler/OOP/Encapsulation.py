"""Encapsulation -> Kapsulleme

Bazi bilgilere bazi kisilerin erismemesi gerekir. Bu yuzden encapsulation kullanabiliriz.
Encapsulation; information hiding yaptigimiz nitelikleri disariya acmamizi saglar.

Information Hiding -> Bilgi Gizleme;
    Class'larin icinde belirlenen (Class Attributes veya Instance Attributes) nitelikleri gizlemeye denir.


Public; public nitelikler herkes tarafindan ulasilabilen niteliklerdir. Syntax olarak herhangi bir sey
eklememize gerek yoktur. Orn; variable

Private; private nitelikler sadece tanimlandigi Class'larda gorulebilirler. Super Class'tan Inheritance
alan Class'lar da bile erisim mumkun degildir. Nitelikleri private tanimlamak icin;
nitelik basina 2 tane underscore getiririz. Orn; __variable

Protected; protected tanimlanan nitelik disariya kapali olur ancak Inheritance alinan Class'larda kullanilir.
Nitelikleri Protected yapmak icin nitelik isminin basina underscore getiririz. Orn; _variable


Getter; Disariya nitelikleri acmak istiyorsak Get metodu ile yapariz.

Setter; Disaridan gelen nitelik degerlerini Set metodu icinde ayarlariz.
"""


class Kullanici(object):
    """Kullanici Super Class'imiz"""
    # Gelistiricilere Bu class'in Parent Class old. belirtmek icin (object) seklinde yazariz.

    def __init__(self, kAdi, parola):
        self.__kAdi = kAdi
        self.__parola = parola

    def giris(self):
        print(f'{self.__kAdi} sisteme giris yapti.')

    # Private olarak tanimladik
    # def __giris(self):
    #     print(f'{self.__kAdi} sisteme giris yapti.')

    # Protected olarak tanimladik
    # def _giris(self):
    #     print(f'{self._kAdi} sisteme giris yapti.')

    # get metodu
    def getKAdi(self):
        return self.__kAdi

    # get metodu
    def getParola(self):
        return self.__parola

    # set metodumuz
    def setKAdi(self, kAdi):
        self.__kAdi = kAdi

    # set metodu
    def setParola(self, parola):
        if(len(parola) < 5):
            print('Parola Guncellenemedi')
        else:
            self.__parola = parola
            print('Parola Basariyla Degistirildi.')


class Yonetici(Kullanici):
    """Yonetici Sub Class'imiz.

    Args:
        Kullanici (class): Inheritance alinan Class.
    """

    def __init__(self, kAdi, parola):
        # ust sinifin constructor'ini 2 sekilde kullanabiliriz
        # super().__init__(kAdi, parola) # Kullanim Sekli - 1
        # veya
        # Kullanim Sekli - 2 ** self yazmak zorundayiz
        Kullanici.__init__(self, kAdi, parola)

    def giris(self):
        print(f'Kullanici-2 {self.__kAdi} giris yapti')


y1 = Kullanici('ali', '0000')
print(y1.getKAdi())  # private degiskene get metodu ile erisim sagladik
y1.setKAdi('Ahmet')  # private degiskene deger atamasi yapiyoruz
print(y1.getKAdi())
y1.setParola('12345')
print(y1.getParola())
