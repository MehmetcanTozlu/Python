"""Closure

Fonksiyonlara esneklik kazandirir.

Geri donus degeri olarak fonksiyon gonderir.
"""


def func1(v1):
    print(v1)

    def func2(v2):
        print(v2)
    return func2 # Closure'da func2() gibi bir kullanim yapmamaliyiz


v = func1('Ali')
v('Veli')
