"""Dosya Islemleri

r (read) -> Dosya okuma modu. Dosya mevcut degilse hata yazdirir.

a (append) -> Dosya guncelleme/veri ekleme modu. Dosya mevcut degilse yeni dosya olusturur.
Dosya mevcutsa icindekileri silmez, uzerine yazar.

w (write) -> Yeni dosya yazma/olusturma modu. Dosya mevcut degilse yeni dosya otomatik olusturur.
Dosya mevcutsa da silip yenisini olusturur ve icindeki veriler gider!

x (create) -> Belirtilen dosya olusturulur. Dosya zaten olusturulmussa hata verir.

####################################################################################
# Binary(Ikili Dosyalar)
rb -> Binary dosyalari okumak icin kullanilir
wb -> Binary dosyalari yazmak icin kullanilir
ab -> Binary dosyalari eklemek icin kullanilir
xb -> Binary dosyalari yazmak icin kullanilir
####################################################################################

r+ - Read(+) -> Dosya Okuma Yazma Modu
w+ - Write(+) -> Dosya Okuma Yazma Modu   Dosya yoksa olusturur
a+ - Append(+) ->  Dosya Okuma Ekleme modu
x+ - Create(+) ->  Dosya Okuma Yazma Modu Ayni isimde dosya varsa hata verir
"""

# Dosya Olusturmak a, w
# with deyiminde f.close() yazilmasina gerek yoktur
with open(file='C:/users/mehmet/desktop/deneme.txt', mode='w', encoding='utf-8') as f:
    f.write('Python\n')
    f.write('Data Science\n')
    f.write('icin\n')
    f.write('harika bir dil\n')
    f.write('Python is for data science to wonderful')

# # Dosya Okumak r
# with open(file='C:/users/mehmet/desktop/deneme.txt', mode='r', encoding='utf-8') as f:
#     print(f.read())
#     print('İmlec: ', f.tell()) # imlecin indexini verir
#     print('Imlecin yeni konumu', f.seek(30)) # imleci istedigimiz index'e goturuyoruz
#     print(f.read())

# # Dosyanin Basina Guncellemek r+
# with open(file='C:/users/mehmet/desktop/deneme.txt', mode='r+', encoding='utf-8') as f:
#     eskiVeri = f.read()
#     f.seek(0) # imleci 0. konuma getirdi oraya ekledi
#     f.write('Merhaba 12345\n' + eskiVeri)

# # Dosyanin Sonuna Guncellemek r+
# with open(file='C:/users/mehmet/desktop/deneme.txt', mode='a', encoding='utf-8') as f:
#     f.write('\nBu son yazdigimiz yazi!')

# # Dosyanin Ortasina Guncelle Yapmak r+
# with open(file='C:/users/mehmet/desktop/deneme.txt', mode='r+', encoding='utf-8') as f:
#     eskiVeri = f.readlines() # okunan degerleri bir listenin elemanlari olarak geri dondurur
#     eskiVeri.insert(1, 'Ortasina eklenen veri\n')
#     f.seek(0) # imleci en basa aldik cunku araya ekledigimiz liste = eskiVeri
#     f.writelines(eskiVeri) # bir dosyaya liste eklememize yarar
