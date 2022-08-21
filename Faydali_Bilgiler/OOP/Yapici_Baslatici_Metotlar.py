"""Yapici ve Baslatici Metotlar

Constructor -> Yapici Metot
Initializer function -> Baslatici Metotlar

Python haricindeki diger prog. dillerinde constructor metodu nesneye ilk degerlerini atamakla sorumludur.
Nesnenin ilk degerleri nesne olusturulurken verilir. Oncesinde de nesneye hafizadan yer ayarlanir.
Bu islem python haricindeki diger prog dillerinde eszamanli yapilir.
Ancak nesneye deger atamasi ve hafizadan nesneye yer ayrilmasi python'da eszamanli olarak gerceklesmez.

Bir nesnenin yasam dongusu;
1- Constructor -> Nesneye bellekten yer tahsis edilmesi
2- Initialization -> Nesneye ilk degerlerinin verilmesi
3- Destruction -> Nesnenin bellekten atilmasi
"""

# Python'da butun siniflar varsayilan olarak object sinifindan miras almistir.


# object yazsakta yazmasakta python object sinifini miras alir.
class Person(object):

    # new metotu varsayilan olarak arkaplanda calisir. Biz burda Override islemi yaptik.
    # new metodunu bu sekil varsayilan olarak degilde degistirseydik __init__ (initializer) metodu calismayacakti
    # __new__ metodunu kesinlikle kullanmamaliyiz, override etmemeliyiz
    # Constructor Method
    def __new__(cls, *args, **kwargs):  # sinif degeri old. icin cls yazmaliyiz
        print('New Metodu')
        return super().__new__(cls, *args, **kwargs)

    # Diger dillerde constructor denilir ancak Python'da initializer metot denir.
    # Yani nesneye ilk degerleri verme metodu
    # Python'da constructor farkli bir anlama gelir. Nesnenin bellekte bir yere yerlesmesi anlamina gelir.
    # Ilk calisan metot constructor metodudur.
    # Instance Method
    def __init__(self):
        print('Nesne Uretildi')


p1 = Person()
