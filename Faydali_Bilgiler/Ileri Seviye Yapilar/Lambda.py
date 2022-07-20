"""Lambda

Lambda fonksiyonlari, isim vermeden kullanabildigimiz fonksiyonlardir.
Cok hizli calisirlar. Birden fazla deger alabilirler.
"""

a = lambda x, y: x + y
b = a(10, 20)
print(b)

x = lambda isim, soyisim: f'Isminiz: {isim}\nSoyisminiz: {soyisim}'
print(x('Ali', 'Candan'))

y = (lambda x, y: x ** y)(2, 4)
print(y)
