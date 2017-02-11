# -*- coding: utf-8 -*-
# Detect face and cutting out it 

import cv2

cascade_path = "~/.pyenv/versions/anaconda3-4.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

def detectFace(image)
    # gray scale for loading image easily
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    # equalizeing for changing image clearly
    image_gray = cv2.equalizeHist(image_gray)

    # using cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    print "face rectangle"
    print facerect

    return facerect

video_path = "movie.mp4"
cap = cv2.VideoCapture(video_path)
framenum = 0
faceframenum = 0
color = (255, 255, 255)

while(1):
    framenum += 1
    ret, image = cap.read()
    if not ret:
        break
    if framenum%10==0:
        facerect = detectFace(image)
        if len(facerect) == 0: continue
        for rect in facerect:
            croped = image[rect[1]]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            cv2.imwrite("detect" + str(faceframenum) + ".jpg", croped)
        faceframe += 1
cap.release()
