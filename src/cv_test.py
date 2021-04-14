import os
import sys
import logging
import random
import json
import numpy as np
import cv2
import PIL
import pathlib
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def main():
    RESIZE =512
    src =
    '/opt/program/services/project/cassava-leaf-disease-classification/test_images/2216849948.jpg'
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img.shape)
    resize_img = cv2.resize(img, (RESIZE, RESIZE))
    print(resize_img.shape)

    plt.imshow(img)
    plt.axis('on')
    plt.title('src_image')
    #plt.show()
    plt.imshow(resize_img,cmap='gray')
    plt.axis('on')
    plt.title('dst_image')
    #plt.show()

    gray_resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
    print(gray_resize_img.shape)
    plt.imshow(gray_resize_img, cmap='gray')
    plt.axis('on')
    plt.title('gray_resize_image')
    plt.show()

if __name__ == '__main__':
    main()
