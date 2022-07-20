"""PEP257 ve PEP8 Standartlarinda Fonksiyon DocString Tanimi"""


def toplama(x: int, y: int) -> int:
    """Bu Fonksiyon Toplama Fonksiyonudur.
    2 tane deger alir ve geriye 2 degerin (int) toplamini dondurur.

    Args:
        x (int): Gelen ilk parametre.
        y (int): Gelen ikinci parametre.

    Returns:
        int: Parametrelerin toplamini dondurur.
    """
    return x + y


def carpma(x: float, y: float = 5.5) -> None:
    """Carpma Fonksiyonu. Aldigi parametreleri carpar.

    Args:
        x (float): Gelen 1. deger.
        y (float): Gelen 2. deger. Defaults to 5.5.
    """
    print(x * y)
