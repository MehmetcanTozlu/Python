"""Abstraction -> Soyutlama

Abstraction; bir varligin ozelliklerini gizleyip islevlerini gostermektir.
Soyut siniflar Sablon siniflardir yani nesnesi uretilemezler.

Soyut siniflar genelde Inheritance alinarak belli basli islevleri o siniflarin yapmasini saglarlar.
Ornegin; Soyut sinifta bulunan bir methodu Soyut sinifi miras alan bir sinif bu methodu override etmek
zorundadir. Direkt kullanamazlar.

Abstract Class'lari kullanirken diger prog. dillerinde abstract anahtar kelimesi gelir. Python'da ise
Abstract Base Class(Abc) modulunu import etmemiz gereklidir.

Python'da bir class'in abstract olmasi icin ABC sinifindan Inheritance almasi gerekir.
Abstract Class'larin en az bir tane soyut methodunun olmasi gereklidir.
Abstract Class'larda normal methodlarda olabilir ancak cok gerek duyulmaz.
"""

from abc import ABC, abstractmethod


class CepTelefonu(ABC):
    """Cep Telefonu Abstract Class

    Args:
        ABC (Abctract Class)
    """

    @abstractmethod
    def mesajGonder(self, mesaj):
        pass

    @abstractmethod
    def aramaYap(self):
        pass

    def islem(self):
        print('Islem')


class mi20(CepTelefonu):
    """mi20 Telefon Markasi Sinifi

    Args:
        CepTelefonu (Abstract Class): Inheritance alinan sinif.
    """
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model

    def mesajGonder(self, mesaj):
        print(f'{self.marka} telefonundan "{mesaj}" gonderildi.')
    
    def aramaYap(self):
        print(f'{self.model} model telefondan arama yapildi.')
    

v1 = mi20('Qwerty', 'asd-2022')
v1.mesajGonder('Selam')
v1.aramaYap()
