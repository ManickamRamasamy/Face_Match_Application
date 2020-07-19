import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file("images/train.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file("images/test.jpg")
img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)

#detect the face location "[0]" to send and fetch the single/first element. Face location will returns four values
facelocation = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]

#Defect face location in rectangle shape
cv2.rectangle(img,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]), (255,0,0),2)

facelocation_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test,(facelocation_test[3],facelocation_test[0]),(facelocation_test[1],facelocation_test[2]), (255,0,0),2)

#Compare the Train image and test image for the test
results = face_recognition.compare_faces([encode],encode_test)
print(results)

#To find accuracy of image by detecting the face distance
facedistance = face_recognition.face_distance([encode],encode_test)
print(facedistance)
cv2.putText(img_test,f'{results},{round(facedistance[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('train',img)
cv2.imshow('test',img_test)
cv2.waitKey(0)