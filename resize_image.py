# -*- coding:utf-8 -*-

from PIL import Image
import os

# Images Directory
imgNames = os.listdir("/Users/YoheitaOnishi/image_recognition/images")

def readImg(imgName):
    try:
        img_src = Image.open("/Users/YoheitaOnishi/image_recognition/images/" + imgName)
        print("read image")
    except:
        print("{} is not image file".format(imgName))
        img_src = 1
    return img_src

for imgName in imgNames:
    img_src = readImg(imgName)
    if img_src == 1:continue
    else:
        resizeImg = img_src.resize((50, 50))
        resizeImg.save("50_50_" + imgName)
        print(imgName + " is done")
