"""Fonksiyonlar
Infiniti(*args, **kwargs) Parameters

*args ve **kwargs arasindaki en onemli fark;
**kwargs fonksiyonunu cagirirken anahtar deger iliskisiyle cagirabilmemizdir. anahar=deger

*args(argumans) -> Tuple
**kwargs(keywords argumans) -> Dict

kisaca *args kullanilan fonksiyon isimsiz parametredir. Tuple'dir.
**kwargs kullanilan fonksiyon isimli parametredir. Key ve Value degerlerinide gondermemiz gerekir. Dict'dir.

kullanimi;
def func(*args):
def func(**kwargs):

Bir fonksiyona ayni anda da 4 farkli yontemde de parametre verebiliriz.
Ancak bunlarin sirasi onemlidir.

def func(arg1, arg2, *args, **kwargs):
"""

print('*args\'a ornek')
def func(*args):
    #Gelen parametreler tuple turunde
    
    print(args)
    # result = 0
    # for i in isimsiz:  # result sum(isimsiz) 'in hazir func kullanmadan yaptik
    #     result += i
    # return result
    return sum(args)


print(func(10, 20, 30, 40, 50))
print('**************************************************************')
print('**kwargs\'a ornek')
def func2(**kwargs):
    # Gelen parametreler sozluktur
    #return isimli
    for k, v in kwargs.items():
        print(f'Anahtar -> {k}\nDeger -> {v}')
        print('-----------------------')


func2(isim='Ali', soyisim='Candan', mesaj='Python')
print('***************************************************************')
print('*args ve **kwargs kullanimi')

def func3(arg1, *args, **kwargs):
    print('Arg1: ', arg1)
    print('Args: ', args)
    print('Kwargs: ', kwargs)


func3(666, 'ali', 'veli', True, 12.12, 55, isim='ali', soyisim='candan', kilo=77.7, yas=39)

