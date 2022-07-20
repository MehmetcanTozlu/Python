"""Reduce Fonksiyonu.

Donguye sokulabilecek herhangi bir veri tipinin icindeki tum elemanlari azaltarak gezen ve karsilastirma
yapmaya imkan taniyan bir yapidir.
"""
from functools import reduce

# Bir ornegi ilk once for kullanarak yapalim daha sonra reduce ile yapalim

result = 1
liste = list(range(1, 10))
for each in liste:
    result *= each
print('Reduce Kullanmadan: ', result)

print('**************************')

result = 1
result = reduce((lambda i, y: i * y), liste)
print('Reduce Kullanarak: ', result)

print('**************************')

# liste icinde en buyuk sayiyi bulma
numbers = [665, 12, 5, 17, -12, 8, 9, 19, 21, -3, 8, -5, 55, 34, 77, -999, 666]
result_2 = reduce((lambda i, y: i if i > y else y), numbers)
print(result_2)
