import cv2

# Video dosyasını yükle
cap = cv2.VideoCapture("Demo Black Line Video for Image Processing.mp4")

while True:
    # Video'dan bir kare al
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gri tonlamalı görüntüyü ikili bir görüntüye dönüştür
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Siyah bantı algılamak için maske oluştur
    mask = cv2.inRange(binary, 0, 50)

    # Maskeyi kullanarak siyah bantı belirle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Siyah bantın merkezini hesapla
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, "Center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Eğer siyah bantın merkezi ekranın solunda ise sola, sağında ise sağa dön
            if cx < frame.shape[1] // 2:
                print("Sola dön")
            else:
                print("Sağa dön")

    # Görüntüyü göster
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırak
cap.release()
cv2.destroyAllWindows()
