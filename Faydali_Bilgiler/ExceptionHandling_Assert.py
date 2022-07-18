"""Hata Yonetimi -> Assert

assert Software Developer'lar icin bir Exception Handling'dir.

Assert(iddia etmek)

Assert'ler derleme sirasinda python -o bayragi ile(konsola -> python -o example.py) devre disi olabilir.

Assert sayesinde, bir kodda her zaman dogru oldugunu varsaydigimiz iddialarda bulunabiliriz.

Assert'in icine bir ifade yazariz ve bu ifade dogruysa kod calismaya devam eder. Ancak ifade yanlissa
raise gibi program coker ve hata/istisna nesnesi doner.
Bu nesneye de AssertionError diyoruz.

**************
Eger kullanici turunde bir hata istiyorsak ve bu hata cozulmedigi muddetce programin cokmesini istiyorsak
raise kullanacagiz. Mesela; liste uzunlugunun 0 olmasi, dosyaya erismek istiyoruz ancak dosya yerinde yok etc.
**************
Eger hatayi yazilimi gelistiren kisilere(developer'lara) vermek istiyorsak bunu assert ile yapiyoruz. Mesela;
projemizde f string var(f'') ve f string python 3.6 surumunden itibaren var. Software Developer'in bizim
kodlarimizi kullanabilmesi icin python 3.6 surumu olmasi gerekli. Burada pythonun surumunu kontrol etmemiz
lazim. Bu gibi durumlarda assert kullaniyoruz.
assert pythonSurumu >= 3.6, 'Lutfen python surumunu 3.6 ve ustune guncelleyin aksi halde fstring calismayacak'
"""

# listede eleman varsa calismaya devam edecek ancak liste bossa hata dondurecek
liste = []


def insert(liste):
    # virgulden oncesi iddia ettigimiz ifade
    # virgulden sonrasi da ifademiz yanlissa calisacak hata nesnesi mesaji
    # eger ifademiz yanlissa assert'ten sonraki hic bir kod blogu calismaz
    assert len(liste) != 0, 'Liste uzunlugu 0...'
    liste.append('Python')
    return liste


try:
    print(insert(liste))

except AssertionError as e:
    print('Error!\n', e)
