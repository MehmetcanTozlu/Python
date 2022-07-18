"""Hata Yonetimi -> Raise

raise kullanicilar icin kullanilan bir Exception Handling'dir.

Raise(artis) genellikle 2 amac icin kullanilir.
1- Python'in diledigi yerde hata verdirmesini saglamak.
2- Bizim diledigimiz yerlerde belirtilen kurallara gore Python tarzinda Exception uretmemizi saglar.

**************
Eger kullanici turunde bir hata istiyorsak ve bu hata cozulmedigi muddetce programin cokmesini istiyorsak
raise kullanacagiz. Mesela; liste uzunlugunun 0 olmasi, dosyaya erismek istiyoruz ancak dosya yerinde yok etc.
**************
Eger hatayi yazilimi gelistiren kisilere(developer'lara) vermek istiyorsak bunu assert ile yapiyoruz. Mesela;
projemizde f string var(f'') ve f string python 3.6 surumunden itibaren var. Software Developer'in bizim
kodlarimizi kullanabilmesi icin python 3.6 surumu olmasi gerekli. Burada pythonun surumunu kontrol etmemiz
lazim. Bu gibi durumlarda assert kullaniyoruz
assert pythonSurumu >= 3.6, 'Lutfen python surumunu 3.6 ve ustune guncelleyin aksi halde fstring calismayacak'
"""

try:
    # 15. satirda ki e deyimi mesaji yerine raise icinde tanimladigimiz hata mesajini dondurecektir
    raise ValueError('Hata olustu!')
    sayi = int(5)
    print(sayi)

except ValueError as e:
    print('Error!\n', e)

print('****************************************************')

sayi = 5
try:
    if sayi == 5:
        raise Exception('Hata olustu!\nLutfen 5 rakamini girmeyin...')
    else:
        print(sayi)

except Exception as e:
    print(e)  # raise icinde girdigimiz mesej dondurur raise ici bossa hic bir sey yazmaz
