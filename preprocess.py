import mindspore as ms
import os
import cv2 as cv

def pre():
    dict = {}
    i = 0
    pic = []
    labellist = []
    for filename in os.listdir('./ImageNet_train'):
        dict[filename] = i
        i += 1

    for filename in os.listdir('./ImageNet_train'):
        for file in os.listdir('./ImageNet_train/' + filename):
            img = cv.imread('./ImageNet_train/' + filename + '/' + file)
            pic.append(img)
            labellist.append(dict[filename])
    return pic, labellist
