import os
import sys
import logging
import random
import json
import time
import math
import numpy as np
import tensorflow as tf
import cv2
import PIL
import pathlib
from pathlib import Path
from train import *

def parse_proto(example):
    feature_dict = {}
    feature_dict["image"] = tf.FixedLenFeature([], tf.string)
    feature_dict["image_name"] = tf.FixedLenFeature([], tf.string)

    struct = tf.parse_single_example(example, features=feature_dict)
    image = tf.image.decode_jpeg(struct["image"])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize_images(image, [384, 384])
    image = tf.reshape(image, [384, 384, 3])

    name = tf.cast(struct["image_name"], tf.string)
    name = tf.reshape(name, [-1])
    return image, name

def online_test():
    root = "/opt/program/services/project/cassava-leaf-disease-classification/"
    model = "%s/model/leaf-disease.model" % root

    path = "%s/test_tfrecords/ld_test00-1.tfrec" % root
    test_dataset = tf.data.TFRecordDataset(path)
    test_dataset = test_dataset.map(parse_proto).repeat(3).batch(1)
    iterator = test_dataset.make_one_shot_iterator()
    image, name = iterator.get_next()

    is_training = tf.placeholder(tf.bool)
    logit, predict, _, _ = inference(image, is_training, True)
    top_one = compute_top_k(predict)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("%s/model" % root)
        if ckpt and ckpt.model_checkpoint_path:
            print(" path: %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        feed_map = {}
        feed_map[is_training] = False
        feed = [image, name, logit, predict, top_one]
        while True:
            try:
                _, ne, lg, pd, indice = sess.run(feed, feed_dict=feed_map)
                print("%s : %s : %s" % (str(ne.tolist()[0]), str(pd.tolist()[0]), str(indice.tolist()[0])))
            except Exception as e:
                #print("%s" % e)
                break

def test():
    root = "/opt/program/services/project/cassava-leaf-disease-classification/"
    model_path = "%s/model/leaf-disease.model" % root
    image_path = Path('%strain_images/' % root)
    tf_records_path = Path('%strain_tfrecords_384_plus_384/' % root)
    all_paths = [str(name) for name in list(tf_records_path.glob('*'))]
    paths = all_paths[-1:]

    batch_size = 32
    queue = tf.train.string_input_producer(paths)
    images, names, labels = gen_batch(queue, batch_size, True)
    labels = tf.reshape(labels, [-1])

    is_training = tf.placeholder(tf.bool)
    use_bn = True
    logits, predicts, _, _ = inference(images, is_training, use_bn)

    label_weights = tf.placeholder(tf.float32)
    loss = compute_loss(logits, labels, label_weights)
    acc = compute_accurency(predicts, labels)
    topk_indices = compute_top_k(predicts)

    true_label = tf.placeholder(tf.int32)
    pred_label = tf.placeholder(tf.int32)
    confusion_matrix = compute_confusion_matrix(true_label, pred_label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("%s/model" % root)
        if ckpt and ckpt.model_checkpoint_path:
            print("check path: %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        test_loss = 0.0
        test_acc = 0.0
        name_set = set()
        it = 0
        count = 0
        hit = 0
        ln = 0

        all_labels = []
        all_indices = []

        feed_map = {}
        feed_map[label_weights] = np.array([1.0] * batch_size, dtype=np.float32)
        feed_map[is_training] = False
        feed = [images, names, labels, logits, predicts, topk_indices, loss, acc]
        try:
            while True:
                _, ns, labs, _, preds, indices, batch_loss, batch_acc = sess.run(feed, feed_dict=feed_map)
                print("correct num: %d, %f" % (ns.size, batch_acc*batch_size))
                ns, labs, indices, preds = ndarray_to_list(ns, labs, indices, preds)
                for e in zip(ns, labs, indices, preds):
                    if e[0] not in name_set:
                        name_set.add(e[0])
                        hit += (e[1] == e[2])
                        all_labels.append(e[1])
                        all_indices.append(e[2])
                name_set.update(ns)
                format_output(ns, labs, indices, preds)
                if len(name_set) <= ln:
                    count += 1
                else:
                    count = 0
                    ln = len(name_set)
                if count == 20:
                    break
                test_loss += batch_loss
                it += 1
        except Exception as e:
            print("exp: %s" % str(e))
        finally:
            test_acc = float(hit/len(name_set))
            print("loss: %.2f, acc: %.2f" % (test_loss, test_acc))
            feed_map.clear()
            feed_map[true_label] = np.array(all_labels)
            feed_map[pred_label] = np.array(all_indices)
            cf_matrix = sess.run(confusion_matrix, feed_dict=feed_map)
            cf_matrix = cf_matrix.tolist()
            print("confusion matrix:")
            for lst in cf_matrix:
                print("{0:<6}{1:<6}{2:<6}{3:<6}{4}".format(lst[0], lst[1], lst[2], lst[3], lst[4]))
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    #online_test()
    test()
