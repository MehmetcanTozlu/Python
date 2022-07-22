"""Name Mangling -> Isim Degistirme

Bu yontem uygulanmamalidir!
Cunku Encapsulation prensibine zıt calisir.
Developerlar bir degiskeni gizliyorsa bir bildigi vardir prensibiyle yaklastigimiz icin varsayilan
yapiyi bozabiliriz.
"""


class Kullanici:
    def __init__(self, kAdi):
        self.__kAdi = kAdi
    
    def __giris(self):
        print('Kullanici Adi: ', self.__kAdi)
    
    def getKAdi(self):
        return self.__kAdi
    
    def setKAdi(self, kAdi):
        self.__kAdi = kAdi


k = Kullanici('Ali')
print(k._Kullanici__kAdi) # private tanimladigimiz giris metoduna ulastik
print('**********************')
k._Kullanici_kAdi = 'Ahmet' # __kAdi private variable'ini degistirdik
print(k._Kullanici_kAdi)
print('**********************')
print(k.getKAdi())
k.setKAdi('Veli')
print(k.getKAdi())
