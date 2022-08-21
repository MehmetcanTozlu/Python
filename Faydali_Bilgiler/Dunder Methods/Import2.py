# Import.py dosyamizda topla() metodumuzu import edelim

# Dogru kullanim.
import Import

Import.topla(10, 20)

# Ayni islemi __import__ ile yapalim

# Dosya isimleri rakam ile basliyorsa bu sekilde ice aktarmak mumkun olur.
# Bu sekilde bir kullanim kesinlikle tavsiye edilmez.
data = __import__('Import')
data.topla(1, 2)
