import pandas as pd

df = pd.read_csv('sepet.csv', header=None)

t = [] # her bir elemanin bir liste olacak sekilde listemizi olusturuyoruz

# apriori algoritmasi girdi olarak dizi icinde dizi seklinde bir list aliyor.
# Bu yuzden verilerimizi bu formata uygun hale donusturelim

for i in range(0, 7501):
    t.append([str(df.values[i, j]) for j in range(0, 20)])

# kaynak: https://github.com/ymoch/apyori
from apyori import apriori

# t = Birliktelik Yapilacak dizinimiz
# min_support = bu degerin altinda kalan veriler elenecek
# min_confidence = minimum guven araligi
# min_lift = minimum kaldirac
# min_length = tablolarimiz minimum 2'li veriler seklinde olusturulsun
rules = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)

print(list(rules))

"""
Cikti:
    
[RelationRecord(items=frozenset({'herb & pepper', 'ground beef'}),
                support=0.015997866951073192,
                ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & pepper'}),
                items_add=frozenset({'ground beef'}),
                confidence=0.3234501347708895,
                lift=3.2919938411349285)]),
 RelationRecord(items=frozenset({'nan', 'herb & pepper', 'ground beef'}),
                support=0.015997866951073192,
                ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & pepper'}),
                items_add=frozenset({'nan', 'ground beef'}),
                confidence=0.3234501347708895,
                lift=3.2919938411349285),
                OrderedStatistic(items_base=frozenset({'nan', 'herb & pepper'}),
                items_add=frozenset({'ground beef'}),
                confidence=0.3234501347708895,
                lift=3.2919938411349285)])]
"""
