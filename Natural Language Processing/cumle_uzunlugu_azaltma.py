sentence = []
with open(file='Final Project2.txt', mode='r', encoding='utf-8') as f:
    sentence.append(f.read().split())

sayac = 0
sayac2 = 6
cumle = ''
derlem = []
for i in sentence[0]:
    cumle = ' '.join(sentence[0][sayac: sayac2])
    derlem.append(cumle)
    cumle = ''
    sayac = sayac2
    sayac2 += 6

with open(file='cumleler.txt', mode='a', encoding='utf-8') as f:
    for i in derlem:
        f.write('{0}\n'.format(i))
