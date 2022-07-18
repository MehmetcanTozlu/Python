from Crypto.Cipher import AES
from secrets import token_bytes

key = token_bytes(16)


def encrypt(msg):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(msg.encode("utf-8"))
    return nonce, ciphertext, tag


def decrypt(nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    try:
        cipher.verify(tag)
        return plaintext.decode("utf-8")
    except:
        return False


nonce, ciphertext, tag = encrypt(input("Sifrelenmesini Istediginiz Mesaji Giriniz: "))
plaintext = decrypt(nonce, ciphertext, tag)
sifreli_metin = ciphertext
print(f"Sifrelenmis Metin: {ciphertext}")

if not plaintext:
    print("Hatali Mesaj!")
#else:
    print(f"Sifrelenecek Metin: {plaintext}")

arr1 = [plaintext]
arr2 = [ciphertext]

with open("Sifrelenen Metnin Bilgisayaraki Yolu", "a", encoding="utf-8") as f:
    # f.write(plaintext + '\n')
    for key in arr2:
        k = str(key).replace("'", "\n")
        if k.find("space") >= 0:
            f.write("\n")
        elif k.find("Key") == -1:
            f.write(k)

print("********************************************")
sifreliMetin = input("Cozmek Istediginiz sifreyi giriniz: ")
#print("sifreliMetin: ", sifreliMetin)
#print("sifreli_metin: ", ciphertext)
if str(sifreliMetin) == str(sifreli_metin):
    print("Sifrelenmis Metin: ", plaintext)
    with open("Cozulmus Sifrenin Bilgisayardaki Yolu", "a", encoding="utf-8") as f:
        for key in arr1:
            k = str(key).replace(" ", "\n")
            if k.find("space") >= 0:
                f.write("\n")
            elif k.find("Key") == -1:
                f.write(k)
else:
    print("Hatali Giris Yaptiniz!")
#print("Sifrelenmis Metin", sifreli_metin) #  ciphertext
#print("Sifresi Cozulmus Metin: ", plaintext)

