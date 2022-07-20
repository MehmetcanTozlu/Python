"""Zip Fonksiyonu.

Birden fazla listeyi indexleri esit olmasi sartiyla birlestirir.
"""

list_1 = [10, 11, 12, 13, 14]
list_2 = ['Pazartesi', 'Sali', 'Carsamba', 'Persembe']
list_3 = ['Temmuz', 'Agustos', 'Ocak', 'Mart']

birlestirilmisListe = list(zip(list_1, list_3, list_2))

def generatorListe(birlestirilmisListe):
    for i in birlestirilmisListe:
        yield i

birlestirilmisListe = iter(birlestirilmisListe)
try:
    print(next(birlestirilmisListe))
    print(next(birlestirilmisListe))
    print(next(birlestirilmisListe))
    print(next(birlestirilmisListe))

except StopIteration as e:
    print(e)
