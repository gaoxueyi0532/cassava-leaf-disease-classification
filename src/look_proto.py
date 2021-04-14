import os
import sys
import logging
import random
import json
import numpy as np
import tensorflow as tf
import cv2
import PIL
import pathlib
from pathlib import Path
from PIL import Image
import gflags
from gflags import *

def main():
    n = tf.random_uniform(shape=[1], maxval=180)
    filenames = ['ld_test00-1.tfrec']
    ds = tf.data.TFRecordDataset(filenames)
    it = ds.make_one_shot_iterator()
    item = it.get_next()
    example = tf.train.Example()
    feature_dict = {}
    feature_dict["image"] = tf.FixedLenFeature([], tf.string)
    feature_dict["image_name"] = tf.FixedLenFeature([], tf.string)
    feature_dict["target"] = tf.FixedLenFeature([], tf.int64)

    rd = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)
    eta = tf.constant(0.5, shape=[1], dtype=tf.float32)
    ans = tf.greater(rd[0],eta[0])
    print(ans.get_shape())
    with tf.Session() as sess:
        ans_out = sess.run(ans)
        print(ans_out)
        return
        while True:
            proto = sess.run(item)
            print(sess.run(example.FromString(proto)))
            break

if __name__ == '__main__':
    main()
