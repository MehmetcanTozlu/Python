"""Static Method

Static Methodlar, ne cls ne de self gibi bir parametre almazlar. Bu yuzden nesne veya siniflarla ilgilenmezler.
Static Methodlari da sinif adlarini yazarak ulasabiliriz. Nesne uretmemize gerek kalmaz. Ancak nesne ile de
erisilebilirler.

Static Methodlar, kendisini cagiran sinif veya ornek hakkinda herhangi bir bilgiye sahip degildir.
Bunlar islevini kaybetmeden, sinif disinda da ayni sekilde tanimlanabilirler.
Class Methodlari ise otomatik olarak kendisini cagiran sinifa veya ornegin sinifina bir referans alir.
"""


class Matematik:

    nesneSayisi = 0

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.nesneEkle()

    def toplama(self):
        return self.x + self.y

    @classmethod
    def nesneEkle(cls):
        cls.nesneSayisi += 1

    @staticmethod
    def PI():
        return 3.14

    @staticmethod
    def max(x, y):
        if x > y:
            return x
        return y


m1 = Matematik(13, 5)
print(m1.toplama())
print(Matematik.nesneSayisi) # m1 nesnesi uretildigi icin 1 doner
print(Matematik.PI())
print(Matematik.max(9, 12))
print(m1.max(28, 11))
