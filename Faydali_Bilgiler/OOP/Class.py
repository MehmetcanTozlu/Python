"""Object Oriented Programming

instance = uretilen nesne
instantiating = nesne uretme islemi
instance attributes = nesne nitelikleri
class attributes = sinif nitelikleri
methods = sinifa ozgu fonksiyonlar
constructor = her sinifta bulunan ve sinif cagrildiginda ilk calisan Yapici Metot
dunder methods = double under(iki alt cizgili) metotlar

OOP daha anlasibilir kod yazmamizi saglar. OOP ile kodlarimiz surdurulebilir olur.
Butunu parcalara ayirdigimiz icin daha hizli ve daha az hata ile ilerlenir.

Class;
Soyut varliklardir.
Kullanilmadikca bellekte yer kaplamazlar.

Object;
Class'lari kullanmak icin bir kopyasi olusturulur. Bu kopyada object'dir. Bu isleme instantiating denir.
Somut varlikdir.
Bellekte yer kaplarlar.
Bir Class'tan istedigimiz kadar object uretebiliriz.
"""


class excampleClass(object):

    # class attributes
    tur = 'Homo Sapiens'

# self; ilgili siniflara veya metotlara ait bilgileri kullanacagimizi belirtmek icin kullaniyoruz
# Uretilen nesnenin referansi self'e aktarilir bu sayede nitelikleri hangi nesnenin kullanacagi bilinir.
# Diger prog. dillerinde self = this

    # constructor metot
    def __init__(self, ad, soyad, boy, kilo):
        """Yapici Metot.
        Bireyin genel ozelliklerini disaridan alir.

        Args:
            ad (str): Kisinin adi.
            soyad (str): Kisinin soyadi.
            boy (str): Kisinin boyu.
            kilo (str): Kisinin kilosu.
        """
        # instance attributes
        self.ad = ad
        self.soyad = soyad
        self.boy = boy
        self.kilo = kilo

    def createPerson(self, bolumu: str, okulu: str) -> str:
        """Kisinin bilgilerini yazdiran metot.

        Args:
            bolumu (str): Kisinin bolumu.
            okulu (str): Kisinin okulu.

        Returns:
            str: Kisinin girdi olarak verdigi tum bilgileri geriye dondurur.
        """
        return \
            f'Adi: {self.ad}\nSoyadi: {self.soyad}\nBoyu: {self.boy}\nKilosu: {self.kilo}\nTuru: {self.tur}\
                \nOkudugu Bolum: {bolumu}\nOkulu: {okulu}'


excClass = excampleClass('Mehmetcan', 'Tozlu', 182, 82)
print(excClass.createPerson('Yazilim Muhendisligi', 'Firat Universitesi'))
