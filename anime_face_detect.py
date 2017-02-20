# -*- coding: utf-8 -*-
# Detect face and cutting out it 

import cv2

cascade_path = "/Users/path/to/lbpcascade_animeface.xml"

def detectFace(image):
    # gray scale for loading image easily
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # equalizeing for changing image clearly
    image_gray = cv2.equalizeHist(image_gray)

    # using cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)
    print(image)
    print(cascade)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    print("face rectangle")
    print(facerect)

    return facerect

video_path = "/Users/path/to/hachikuji_mayoi.mp4"
cap = cv2.VideoCapture(video_path)
framenum = 0
faceframenum = 0
# Initialize with white color
color = (255, 255, 255)

while(cap.isOpened()):
    framenum += 1
    ret, image = cap.read()
    if not ret:
        break
    if framenum%50==0:
        facerect = detectFace(image)
        if len(facerect) == 0: continue
        for rect in facerect:
            croped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            cv2.imwrite("detect" + str(faceframenum) + ".jpg", croped)
        faceframenum += 1
cap.release()
