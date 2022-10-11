import pandas as pd

df = pd.read_csv('Kisa_Veriler.csv', error_bad_lines=False, header=None)

from nltk.tokenize import word_tokenize

import re

arr = []
for i in range(len(df[0])):
    word = word_tokenize(df[0][i])
    word = re.sub('[^a-zA-Z]', ' ', df[0][i])
    word = word.lower()
    word = word.split()
    arr.append(word)

with open(file='Veri_Seti.csv', mode='a', encoding='utf-8') as f:
    for i in arr:
        f.write('{0}\n'.format(i))
