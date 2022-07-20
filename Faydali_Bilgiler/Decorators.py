"""Decorators

Decoratorler Meta Programlamaya ornektir.
Meta Programlama; Yazilan bir programin baska bir program uretmesi veya program yazildigi zaman belli
kod parcasinin kodunun degistirilmesi.

Decoratorler fonksiyonlara cesitli ozellikler kazandirmamizi saglarlar.
"""


def decoratorFunc(incomingFunc):
    print('External Func')

    def innerFunc():
        print('Internal Func')
        incomingFunc()
    return innerFunc

@decoratorFunc
def thirdyFunc():
    print('3. Function')


thirdyFunc()
