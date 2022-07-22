"""Property.

fget: metodu isaret eder
fset: metodu isaret eder
fdel: metodu silmeyi isaret eder
doc: yorum/aciklamayi isaret eder

Syntactic Sugar -> Biz Developerlarin daha kolay kod yazmamiza izin veren bir sozdizimidir.
Bu sozdizimi arkaplanda sistemin nasil calismasini bilmeme luksunu tanir. Bu yapi sadece Python ile sinirli
degildir.
"""


class Kullanici:
    __kAdi = None

    def __init__(self, kAdi):
        # nesneye gonderilen ilk degeride bu sayede kontrol edebiliyoruz
        self.kAdi = kAdi

    @property
    def kAdi(self): # genelde method ismine private degisken ismini veririz
        return self.__kAdi

    @kAdi.setter # buradaki isim ile(kAdi) set methodunun da ismi ayni olmalidir.
    def kAdi(self, kAdi):
        if(len(kAdi) > 3):
            self.__kAdi = kAdi
        else:
            raise ValueError('Kullanici Adi 3 Karakterden Buyuk Olmalidir!')

    # def getKAdi(self):
    #     return self.__kAdi

    # def setKAdi(self, kAdi):
    #     if(len(kAdi) > 3):
    #         self.__kAdi = kAdi
    #     else:
    #         raise ValueError('Kullanici Adi 3 Karakterden Buyuk Olmalidir!')

    # k_adi = property(getKAdi, setKAdi)


k = Kullanici('Osman')
k.kAdi = 'Veli'
print(k.kAdi)
# k.setKAdi('Mehmet')
# print(k.getKAdi())
