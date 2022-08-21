"""Alternative Consturctor

Ornegin; kullanici girisli bir uygulama yaptigimizi dusunelim.
Kullanici sisteme giris yaparken farkli alternatifleri olabilsin.
Istegine gore mail, kullanici adiyla, telefon gibi bilesenlerle giris yapabilsin.
"""

class Giris:
    def __init__(self, arg0, arg1):
        self.arg0 = arg0
        self.arg1 = arg1
    
    # Alternative Constructor
    @classmethod
    def mail(cls, mail, parola):
    # kullanici mail ile giris yapmak isterse consturctorin arg0 parametresine mail,
    # arg1 parametresine parolayi gonderecek
        return cls(mail, parola)
    
    # Alternative Constructor
    @classmethod
    def kAdi(cls, kAdi, parola):
        return cls(kAdi, parola)
    
    # Alternative Constructor
    @classmethod
    def telefon(cls, telefon, parola):
        return cls(telefon, parola)


k1 = Giris('mehmet', '12345')
print(k1.arg0, k1.arg1)
print('--------------------------')
k2 = Giris.mail('abc@abc.mail', '123123')
print(k2.arg0, k2.arg1)
print('--------------------------')
k3 = Giris.kAdi('qwerty', 'asdf')
print(k3.arg0, k3.arg1)
print('--------------------------')
k4 = Giris.telefon('55555555', '55555')
print(k4.arg0, k4.arg1)
