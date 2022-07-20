"""Map Fonksiyonu.

map(function, iterable) seklinde kullanilir.

Elimizdeki fonksiyona datanin her birini tek tek gonderir ve sonucu tek bir obje olarak geri dondurur.
iterable'ler liste, tuple etc. olabilir.
"""

liste = list(range(1, 15))

#liste = iter(liste)

def square(inComingValue):
    return inComingValue ** 2

result = list(map(square, liste))
print(result)

text = ['anTalYA', 'HATAy', 'BursA', 'pYThOn', 'progRamlAMA', 'diLi']

result_2 = list(map(lambda i: i.title(), text))
print(' '.join(result_2))
