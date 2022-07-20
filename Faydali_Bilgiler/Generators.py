"""Generatorler

Tek kullanimliklardir. Bir kere olusturulduktan sonra Hafizadan hemen unutulur.

yield ile kullanilir. yield'da return gibidir deger dondurur ancak yield'dan sonra fonksiyondan cikis yapilmaz.

Bir listeyi for ile islem yaptigimizda python butun islemleri ayni anda yapmaya calisir ve hafizayi yorar.
Generator'lerde ise degerler tek tek uretildigi icin ve hemen sonra hafizadan silindigi icin daha fazla
tasarruf ederiz.

Generator kullanalim kullandiralim!
"""


def func():
    i = 0
    yield i

    i += 2
    yield i

    i = i ** 5
    yield i


x = func()
print(x)
print(type(x))

x = iter(x)
print(next(x))
print(next(x))
print(next(x))
# print(next(x))  # StopIteration Exception'u Throw eder.
print('*************************')

liste = range(1, 11)


def generator(liste):
    for i in liste:
        yield i ** 3


arr = generator(liste)

for i in arr:
    print(i, end=' ')
