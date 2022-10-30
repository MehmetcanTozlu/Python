import cv2
import numpy as np

img=cv2.imread('mermer.jpg')

img=cv2.resize(img,(800,800))

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# kırmızı renk için HSV değerleri
low_red=np.array([170, 50, 70])
high_red=np.array([180, 255, 255])
 

mask = cv2.inRange(hsv_img, low_red, high_red)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
mask = cv2.GaussianBlur(mask, (3, 3), 0)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2]


for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    
    if w > 50: # Resimdeki küçük yerleri tespit etmesin diye. Bu değer resimden resime değişir
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


cv2.imshow("Resim", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

