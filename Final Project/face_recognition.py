import datetime as dt
import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

file = open("test.txt","w")
#name = 0
while(True):
    dt.datetime.now()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    
    for (x, y, w, h) in faces:
        
        now = dt.datetime.now()
        date_string = now.strftime("%B %d, %Y %H:%M")
        detection_date = ("Face Detected on " + date_string + "\n")
        file.write(detection_date)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
         
#        img_item = 'julio%s.png' % name
#        cv2.imwrite(img_item, roi_gray)
        
        color = (0,255,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
#        name += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
file.close() 
cap.release()
cv2.destroyAllWindows()