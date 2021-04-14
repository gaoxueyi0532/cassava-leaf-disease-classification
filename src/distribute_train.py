import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import logging
import random
import json
import time
import math
import numpy as np
import tensorflow as tf
import cv2
import pathlib
from pathlib import Path

def read_and_decode(example):
    with tf.variable_scope("parse", reuse=True):
        feature_dict = {}
        feature_dict["image"] = tf.io.FixedLenFeature([], tf.string)
        feature_dict["image_name"] = tf.io.FixedLenFeature([], tf.string)
        feature_dict["target"] = tf.io.FixedLenFeature([], tf.int64)
        parsed_example = tf.io.parse_single_example(example, features=feature_dict)

        image = tf.io.decode_jpeg(parsed_example["image"])
        image = tf.reshape(image, [256, 256, 3])
        image = tf.dtypes.cast(image, tf.float32)

        image = data_augmentation(image)

        name = tf.dtypes.cast(parsed_example["image_name"], tf.string)
        name = tf.reshape(name, [-1])

        label = tf.dtypes.cast(parsed_example["target"], tf.int32)
        label = tf.reshape(label, [-1])
        return image, name, label


def data_augmentation(image):
    # random flip
    image = tf.image.random_flip_left_right(image)

    # brightness adjustation
    image = tf.image.adjust_brightness(image, delta=random.uniform(-0.2,0.2))

    # hue adjustation
    image = tf.image.adjust_hue(image, delta=random.uniform(-0.2,0.2))

    # normalization
    image = tf.image.per_image_standardization(image)

    # add noise
    shape = tf.shape(image)
    noise = tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    image = image + noise
    image = tf.clip_by_value(image, 0, 1.0)
    return image


def histogram(scope, inputs):
    return tf.summary.histogram(scope, inputs)


def get_label_weights(labels, \
                      predicts, \
                      class_num, \
                      per_label_ratio, \
                      eta=2.0):
    with tf.variable_scope("label_weights", reuse=True):
        # transfer to one-hot format
        onehot_labels = tf.one_hot(labels, class_num)
        onehot_labels = tf.squeeze(onehot_labels, axis=1)
        onehot_labels = tf.dtypes.cast(onehot_labels, tf.float32)

        # base Focal-loss: (1 - predict) ** eta
        output = tf.math.multiply(onehot_labels, predicts)
        predict = tf.math.reduce_sum(output, 1)
        ones = tf.ones(tf.shape(predict), dtype=tf.float32)
        diff = tf.math.subtract(ones, predict)
        weights = tf.map_fn(lambda x:tf.pow(x,eta), diff)

        # enforment Focal-loss: put label percent into consider
        class_weights = tf.map_fn(
                lambda x:tf.reduce_sum(tf.math.multiply(per_label_ratio, x)), \
                onehot_labels)
        sample_weights = tf.math.multiply(class_weights, weights)
        return sample_weights

def compute_loss(logits, labels, loss_weights, batch_size):
    with tf.variable_scope("loss", reuse=True):
        one_hot_labels = tf.one_hot(labels, 5)
        one_hot_labels = tf.squeeze(one_hot_labels, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, \
                                               logits=logits, \
                                               weights=loss_weights, \
                                               label_smoothing=0.1)
        return tf.math.reduce_sum(loss) / batch_size


def format_output(names, ys, ys_, predicts):
    names, labels, indices, predicts = ndarray_to_list(names, ys, ys_, predicts)
    print("{0:<20}{1:<10}{2:<10}{3}".format("Name","Label","Predict","Prob"))
    for e in zip(names, ys, ys_, predicts):
        print("{0:<20}{1:<10}{2:<10}{3:.2f}".format(e[0],e[1],e[2],e[3]))


def ndarray_to_list(names, labels, indices, predicts):
    batch_name = []
    batch_label = []
    batch_indice = []
    batch_predict = []
    for g in zip(names, labels, indices, predicts):
        batch_name.extend(g[0].tolist())
        batch_label.extend(g[1].tolist())
        batch_indice.extend(g[2].tolist())
        batch_predict.extend(g[3].tolist())
    return batch_name, batch_label, batch_indice, batch_predict


def compute_accurency(predict, label):
    with tf.variable_scope("acc", reuse=True):
        pred = tf.dtypes.cast(tf.math.argmax(predict, 1), dtype=tf.int32)
        correct_pred = tf.math.equal(pred, label)
        accuracy = tf.math.reduce_sum(tf.dtypes.cast(correct_pred, tf.float32))
        return accuracy


def compute_top_k(predict):
    with tf.variable_scope("topk", reuse=True):
        indices = tf.math.top_k(predict)
        return indices


def Summary(key, value):
    return tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value),])


def clear_log(train_dir, val_dir):
    train_files = os.listdir(train_dir)
    for tf in train_files:
        os.remove(os.path.join(train_dir, tf))
    val_files = os.listdir(val_dir)
    for vf in val_files:
        os.remove(os.path.join(val_dir, vf))

def train():
    root = "/gaoxueyi"
    model_path = "%s/model/leaf-disease.model" % root

    train_log_dir = root + "/src/summary_log/train/"
    val_log_dir = root + "/src/summary_log/eval/"
    clear_log(train_log_dir, val_log_dir)

    label_to_disease_dict = {}
    with open(root + "/data/label_num_to_disease_map.json") as f:
        label_to_disease_dict = json.loads(f.readline().strip())

    name_to_label_dict = {}
    per_label_num = {}
    n = 0
    with open(root + "/data/train.csv") as f:
        for line in f.readlines():
            n += 1
            name, label = line.strip().split(',')
            name_to_label_dict[name] = int(label)
            key = int(label)
            per_label_num[key] = per_label_num.get(key,0) + 1

    # label_num / total
    class_num = len(label_to_disease_dict)
    label_ratio = [float(per_label_num[i] / n) for i in range(class_num)]
    per_label_ratio = tf.constant(label_ratio, dtype=tf.float32)

    # read tfrecords data
    tf_records_path = Path('%s/train_tfrecords_256_plus_256/' % root)
    all_paths = [str(name) for name in list(tf_records_path.glob('*'))]
    file_num = len(all_paths)

    # train, validation data split
    train_paths = all_paths[0 : int(file_num*0.7)+2]
    eval_paths = all_paths[int(file_num*0.7)+2 : int(file_num*0.9)+1]

    epoch_num = 10
    batch_size = 32
    buffer_size = 1024

    # one kind of train strategy: single node, multi-GPU/CPU, sync update
    #device_list = ["/device:CPU:0", "/device:XLA_CPU:0", "/device:XLA_GPU:0", "/device:GPU:0"]
    #device_list = ["/device:CPU:0", "/device:GPU:0"]

    strategy = tf.distribute.MirroredStrategy()
    print(strategy.num_replicas_in_sync)

    with strategy.scope():
        train_dataset = tf.data.TFRecordDataset(train_paths)
        train_dataset = train_dataset.map(read_and_decode).repeat(epoch_num).shuffle(buffer_size).batch(batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        train_iterator = train_dist_dataset.make_initializable_iterator()
        #train_iterator = train_dist_dataset.make_one_shot_iterator()

        eval_dataset = tf.data.TFRecordDataset(eval_paths)
        eval_dataset = eval_dataset.map(read_and_decode).repeat(1).batch(batch_size)
        eval_dist_dataset = strategy.experimental_distribute_dataset(eval_dataset)
        eval_iterator = eval_dist_dataset.make_initializable_iterator()
        #eval_iterator = eval_dist_dataset.make_one_shot_iterator()

        is_training = tf.placeholder(tf.bool)
        global_step = tf.train.get_or_create_global_step()
        num_batch_per_epoch = int(len(name_to_label_dict) / batch_size)
        decay_steps = num_batch_per_epoch * 3

        def batch_norm(inputs):
             norm_outputs = tf.contrib.layers.layer_norm(inputs)
             return norm_outputs

        def conv2D(is_training, \
                   inputs, \
                   filters, \
                   kernal_size=[3,3], \
                   padding='valid', \
                   strides=1, \
                   kernal_initializer=tf.initializers.he_normal(), \
                   activation=tf.nn.leaky_relu, \
                   use_batch_norm=False):
            conv2d_layer = tf.layers.Conv2D(filters=filters,\
                                            kernel_size=kernal_size,\
                                            padding=padding,\
                                            strides=strides,\
                                            kernel_initializer=kernal_initializer)
            outputs = conv2d_layer(inputs)
            if use_batch_norm:
                outputs = batch_norm(outputs)
            if activation:
                outputs = activation(outputs)
            scope_name = tf.get_variable_scope().name
            conv2d_kernal = tf.get_variable("conv2d/kernel", \
                    shape=[kernal_size[0], kernal_size[1], inputs.get_shape()[-1], filters])
            print("%s output shape: %s" % (scope_name, str(outputs.get_shape())))
            return (outputs, conv2d_kernal)

        def max_pooling_2D(inputs, \
                           pool_size=[2,2], \
                           strides=2):
            output = tf.layers.max_pooling2d(inputs=inputs, \
                                             pool_size=pool_size, \
                                             strides=strides)
            scope_name = tf.get_variable_scope().name
            print("%s output shape: %s" % (scope_name, str(output.get_shape())))
            shape = output.get_shape()
            size = shape[1] * shape[2] * shape[3]
            return output, size
        
        
        def dense(is_training, \
                  inputs, \
                  units, \
                  activation=tf.nn.leaky_relu, \
                  kernel_initializer=tf.initializers.he_normal(), \
                  use_batch_norm=False):
            dense_layer = tf.layers.Dense(units=units,\
                                          kernel_initializer=kernel_initializer)
            dense_output = dense_layer.apply(inputs)
            if use_batch_norm:
                dense_output = batch_norm(dense_output)
            if activation:
                dense_output = activation(dense_output)
            scope_name = tf.get_variable_scope().name
            dense_kernal = tf.get_variable("dense/kernel",
                    shape=[inputs.get_shape()[-1], dense_output.get_shape()[-1]])
            print("%s output shape: %s" % (scope_name, str(dense_output.get_shape())))
            return (dense_output, dense_kernal)
        
        
        def dropout(is_training, inputs, rate):
            dropout = tf.layers.dropout(inputs=inputs, \
                                        rate=rate, \
                                        training=is_training)
            scope_name = tf.get_variable_scope().name
            print("%s output shape: %s" % (scope_name, str(dropout.get_shape())))
            return dropout

        def inference(images, is_training, use_bn=False):
            W_list = []
            A_list = []
            kwargs = {}
            with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                kwargs["inputs"] = images
                kwargs["filters"] = 64
                kwargs["strides"] = 2
                kwargs["kernal_size"] = [5,5]
                kwargs["is_training"] = is_training
                kwargs["use_batch_norm"] = use_bn
                x, w = conv2D(**kwargs)
                #W_list.append(histogram("conv1_weights", w))
                #A_list.append(histogram("conv1_activate", x))

            with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
                kwargs["inputs"] = x
                kwargs["filters"] = 64
                kwargs["strides"] = 1
                kwargs["kernal_size"] = [3,3]
                x, w = conv2D(**kwargs)
                #W_list.append(histogram("conv2_weights", w))
                #A_list.append(histogram("conv2_activate", x))

            with tf.variable_scope("pooL1", reuse=tf.AUTO_REUSE):
                x, _ = max_pooling_2D(x)

            with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
                kwargs["inputs"] = x
                kwargs["filters"] = 128
                kwargs["padding"] = 'same'
                x, w = conv2D(**kwargs)
                #W_list.append(histogram("conv3_weights", w))
                #A_list.append(histogram("conv3_activate", x))

            with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE):
                kwargs["inputs"] = x
                kwargs["filters"] = 128
                x, w = conv2D(**kwargs)
                #W_list.append(histogram("conv4_weights", w))
                #A_list.append(histogram("conv4_activate", x))

            with tf.variable_scope("pooL2", reuse=tf.AUTO_REUSE):
                x, _ = max_pooling_2D(x)

            with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE):
                kwargs["inputs"] = x
                kwargs["filters"] = 256
                x, w = conv2D(**kwargs)
                #W_list.append(histogram("conv5_weights", w))
                #A_list.append(histogram("conv5_activate", x))

            with tf.variable_scope("pooL3", reuse=tf.AUTO_REUSE):
                x, size = max_pooling_2D(x)
                x = tf.reshape(x, [-1,size])
        
            with tf.variable_scope("drop1", reuse=tf.AUTO_REUSE):
                x = dropout(is_training, x, 0.2)
        
            with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
                x, w = dense(is_training, x, 1024, use_batch_norm=use_bn)
                #W_list.append(histogram("fc1_weights", w))
                #A_list.append(histogram("fc1_activate", x))
        
            with tf.variable_scope("drop2", reuse=tf.AUTO_REUSE):
                x = dropout(is_training, x, 0.5)
        
            with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
                x, w = dense(is_training, x, 1024, use_batch_norm=use_bn)
                #W_list.append(histogram("fc2_weights", w))
                #A_list.append(histogram("fc2_activate", x))
        
            with tf.variable_scope("fc3", reuse=tf.AUTO_REUSE):
                x, w = dense(is_training, \
                             inputs=x, \
                             units=5, \
                             activation=None, \
                             kernel_initializer=tf.glorot_uniform_initializer())
                #W_list.append(histogram("fc3_weights", w))
                #A_list.append(histogram("fc3_activate", x))
        
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                predict = tf.nn.softmax(x)
            return x, predict, W_list, A_list

        def gradient_descent(global_step, decay_steps, loss):
            lr = tf.train.exponential_decay(learning_rate=0.01,\
                                            global_step=global_step,\
                                            decay_steps=decay_steps,\
                                            decay_rate=0.1,\
                                            staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=0.05)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grad = opt.compute_gradients(loss)
                for i, (g,v) in enumerate(grad):
                    if g is not None: grad[i] = (tf.clip_by_norm(g,4),v)
                apply_grad_op = opt.apply_gradients(grad, global_step)
            return apply_grad_op

        def train_step(inputs):
            images, names, labels = inputs
            logits, predicts, _, _ = inference(images, is_training)

            indices = compute_top_k(predicts)

            # compute loss
            label_weights = get_label_weights(labels, predicts, class_num, per_label_ratio)
            loss = compute_loss(logits, labels, label_weights, batch_size)

            # compute acc
            acc = compute_accurency(predicts, labels)

            # update gradients
            grad_opt = gradient_descent(global_step, decay_steps, loss)
            return (loss, acc, names, labels, predicts, indices)

        def eval_step(inputs):
            images, names, labels = inputs
            logits, predicts, _, _ = inference(images, is_training)
            indices = compute_top_k(predicts)
            label_weights = tf.constant([1.0] * class_num, dtype=tf.float32)
            loss = compute_loss(logits, labels, label_weights, batch_size)
            acc = compute_accurency(predicts, labels)
            return (loss, acc, names, labels, predicts, indices)

        def distribute_train(inputs):
            res = strategy.experimental_run_v2(train_step, args=(inputs,))
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, res[0])
            acc = strategy.reduce(tf.distribute.ReduceOp.SUM, res[1])
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('acc', acc)
            names = strategy.experimental_local_results(res[2])
            labels = strategy.experimental_local_results(res[3])
            predicts = strategy.experimental_local_results(res[4])
            indices = strategy.experimental_local_results(res[5])
            return (loss, acc, names, labels, predicts, indices)

        def distribute_eval(inputs):
            res = strategy.experimental_run_v2(eval_step, args=(inputs,))
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, res[0])
            acc = strategy.reduce(tf.distribute.ReduceOp.SUM, res[1])
            names = strategy.experimental_local_results(res[2])
            labels = strategy.experimental_local_results(res[3])
            predicts = strategy.experimental_local_results(res[4])
            indices = strategy.experimental_local_results(res[5])
            return (loss, acc, names, labels, predicts, indices)
        
        train_next_batch = train_iterator.get_next()
        eval_next_batch = train_iterator.get_next()
        train_res = distribute_train(train_next_batch)
        eval_res = distribute_eval(eval_next_batch)

        #ops = tf.get_default_graph().get_operations()
        #bn_update_ops = []
        #for x in ops:
        #    if ("AssignMovingAvg" in x.name) and (x.type=="AssignSubVariableOp"):
        #        bn_update_ops.append(x)
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn_update_ops)

        def evaluate(sess, feed_map):
            total_loss = 0
            total_acc = 0
            counter = 0
            num = 0
            try:
                while True:
                    eval_loss, eval_acc, names, labels, predicts, indices = sess.run(eval_res, feed_dict=feed_map)
                    format_output(names, labels, predicts, indices)
                    total_loss += eval_loss
                    total_acc += eval_acc
                    counter += 1
                    num = counter * batch_size
            except:
                pass
            return (total_loss, total_acc / num)


        conf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=conf) as sess:
            train_writer = tf.summary.FileWriter("./summary_log/train", graph=sess.graph)
            eval_writer = tf.summary.FileWriter("./summary_log/eval", graph=sess.graph)

            saver = tf.train.Saver(allow_empty=True)
            sess.run(tf.initializers.global_variables())
            sess.run(tf.initializers.local_variables())
            sess.run(train_iterator.initializer)
            sess.run(eval_iterator.initializer)
            try:
                count = 0
                history_train_loss = []
                history_train_acc = []
                feed_map = {}
                while True:
                    count += 1
                    feed_map[is_training] = True
                    print("gxy16")
                    train_loss, train_acc, names, labels, predicts, indices = sess.run(train_res, feed_dict=feed_map)
                    print("gxy17")
                    history_train_loss.append(train_loss)
                    history_train_acc.append(train_acc)

                    print("gxy18")
                    format_output(names, labels, indices, predicts)
                    print("gxy19")

                    if count % int(num_batch_per_epoch * 0.1) == 0:
                        print("gxy20")
                        train_ave_acc = float(sum(history_train_acc) / (count*batch_size))
                        train_ave_acc_suy = Summary("acc", train_ave_acc)
                        train_writer.add_summary(train_ave_acc_suy, count)
                        #train_ave_loss = float(sum(history_train_loss) / (count*batch_size))
                        train_ave_loss = float(sum(history_train_loss) / count)
                        train_ave_loss_suy = Summary("loss", train_ave_loss)
                        train_writer.add_summary(train_ave_loss_suy, count)
                        print("gxy21")
                        print('train step: {}, loss: {}, acc: {}'.format(count, train_ave_loss, train_ave_acc))
                        saver.save(sess, model_path, global_step=count, var_list=tf.global_variables())
                    #if count % int(num_batch_per_epoch * 0.1) == 0:
                    if count <= 5:
                        print("gxy22")
                        feed_map[is_training] = False
                        eval_loss, eval_acc = evaluate(sess, feed_map)
                        print("gxy23")
                        eval_loss_suy = Summary("loss", eval_loss)
                        eval_acc_suy = Summary("acc", eval_acc)
                        eval_writer.add_summary(eval_loss_suy, count)
                        eval_writer.add_summary(eval_acc_suy, count)
                        print("gxy24")
                        print('eval step: {}, loss: {}, acc: {}'.format(count, eval_loss, eval_acc))
            except Exception as e:
                print("train exp: %s" % str(e))
            finally:
                print("finish training")
                train_writer.close()
                eval_writer.close()

if __name__ == '__main__':
    train()
