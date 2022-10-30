import cv2

kamera= cv2.VideoCapture(0)

while True:
    ret,kare=kamera.read()
    kare=cv2.resize(kare,(300, 300))

    yuz=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    goz=cv2.CascadeClassifier("haarcascade_eye.xml")
    gri=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    yuzler=yuz.detectMultiScale(gri,1.1,4)

    for (x,y,w,h,) in yuzler:
        yeni=kare[y:y+h//2,x:x+w]
        gri_yeni=cv2.cvtColor(yeni,cv2.COLOR_BGR2GRAY)
        gozler=goz.detectMultiScale(gri_yeni,1.1,4)
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),2)
        for (a,b,c,d) in gozler:
            cv2.rectangle(yeni,(a,b),(a+c,b+d),(0,255,0),2)

    cv2.imshow("kamera",kare)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
