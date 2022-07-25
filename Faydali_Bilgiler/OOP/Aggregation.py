"""Aggregation -> Birlestirme

Has-A iliskisi/bagi vardir. Ancak bu bag Aggregation'da zayiftir.

Is-Part-Of Relation

Iki sinif arasinda birbirinin parcasi olma anlami vardir.

Ornegin; fakulte ve akademisyenleri dusunelim. Fakulteler ortadan kalkarsa akademisyenler varliklarini
surdurebilir. Yani biri yok old. digeri yok olmaz!
"""


class Akademisyen:

    akademisyenNo = 1456

    def __init__(self, akademisyenAdi) -> None:
        self.akademisyenAdi = akademisyenAdi

    def __repr__(self) -> str:
        return 'Akademisyen Adi: ' + str(self.akademisyenAdi)


class Bolum:
    def __init__(self, bolumAdi) -> None:
        self.bolumAdi = bolumAdi
        # Composition'da burada nesne olusturuyorduk ve o sinifin nesnesini sildigimizde
        # bu sinifta yok oluyordu

    def akademisyenEkle(self, akademisyenAdi) -> None:
        # akademisyen'in Akademisyen turunden old. belirledik
        # akademisyen bir nevi referans tutucudur
        self.akademisyen: Akademisyen = akademisyenAdi

    def __repr__(self) -> str:
        return 'Bolum Adi: ' + str(self.bolumAdi) + \
            '\n' + str(self.akademisyen)


a1 = Akademisyen('Mehmet Yilmaz')
b1 = Bolum('Yazilim Muhendisligi')

# Aggregation Islemini Yapalim
b1.akademisyenEkle(a1)
print(b1)
# del(b1) # b1 yok olmasina ragmen a1'in niteliklerine ulasabiliriz buda zayif iliskiyi gosterir

print(b1.akademisyen.akademisyenNo)
