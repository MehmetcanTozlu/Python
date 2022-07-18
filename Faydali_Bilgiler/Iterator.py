"""Iteratorler

Iterator = Sayilabilen sayida deger iceren nesne
Iterable = Gezilebilen nesne

Iterable olan bir yapiyi sadece 1 kez gezebiliriz.
"""

sayilar = [1, 2, 3, 4]

sayilar = iter(sayilar)

# print(next(sayilar))
# print(next(sayilar))
# print(next(sayilar))
# print(next(sayilar))
# # print(next(sayilar)) Dizinin 5. elemani olmadigindan StopIteration Exception'u Throw etti

while True:
    try:
        print(next(sayilar))
    except StopIteration as e:
        break
