"""Filter Fonksiyonu.

filter(function, iterable) seklinde kullanilabilir.

Adindan da anlasilacagi gibi iterable nesneleri filtrelemek icin kullanilabilir.
"""

result = []
liste = ['Ali', 'Ayşe', 'Mehmet', 'Cengiz', 'Hülya', 'İlhan', 'Onur', 'Zeynep', 'Bulut', 'Betül']
arr = 'Ali Mehmet Betül Müslüm Hüseyin Ahmet Onur'

result = list(filter(lambda i: i in arr, liste))
print(result)

liste_2 = list(range(1, 51))
result = []
result = list(filter(lambda i: i % 9 == 0, liste_2))
print(result)
