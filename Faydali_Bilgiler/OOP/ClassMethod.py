"""Class Method

class attributes'ler birer instance attributes olarak kullanilabilir ancak instance attributes'ler
class attributes olarak kullanilamazlar.
"""

class Personel:

    # Class Attributes
    personelSayisi = 0

    # instance Methods
    def __init__(self, ad: str) -> None:
        # instance Attributes
        self.ad = ad
        self.ekle()

    # instance Methods
    def bilgi(self):
        print('Personel: ', self.ad)

    # class Methonds
    @classmethod # Bu sayede direkt class ismiyle ulasabiliriz
    def ekle(cls):
        cls.personelSayisi += 1


p1 = Personel('Ali')
p2 = Personel('Ahmet')
p3 = Personel('Ayse')

print(Personel.personelSayisi)

p1.bilgi()
print(p1.personelSayisi)
