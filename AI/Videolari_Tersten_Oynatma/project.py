import cv2

camera = cv2.VideoCapture('video ismi/uzantisi.mp4') # oynatilacak videomuzu yukluyoruz

kaydedici = cv2.VideoWriter_fourcc(*'XVID') # kaydedicinin formati

# tersi isminde 30 fps olacak sekilde kaydediciyi ayarladik.
# 1920, 1080 videonun orijinal boyutu. Aksi halde baska bir deger girersek hata veya bos goruntu alabiliriz.
cikis = cv2.VideoWriter('tersinin ismi.avi', kaydedici, 30, (1920, 1080))

x = True
arr = []
while x:
    x, kare = camera.read()
    print('Video Islem Icin Hazirlaniyor...')
    if x:
        arr.append(kare)

y = len(arr)

for i in range(y):
    yeni = x[y - i - 1]
    cikis.write(yeni)
    print('Yeni Video Olusturuluyor...')

camera.release()
cv2.destroyAllWindows()

