from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import project_setup

import os
import sys
from glob import glob
import random
import numpy as np
import tensorflow as tf
import time
import psmlearn 
from psmlearn.vgg16 import vgg16
import preprocess

def add_loss(vgg, labels_placeholder, num_outputs, args):
    with tf.name_scope('fc3'):
        W = tf.Variable(tf.truncated_normal([4096, num_outputs],
                                            dtype=tf.float32, stddev=1e-1),
                        name='weights', trainable=True)
        B = tf.Variable(tf.constant(1.0, shape=[num_outputs], dtype=tf.float32), trainable=True, name='biases')
        vgg.logits = tf.nn.bias_add(tf.matmul(vgg.fc2, W), B, name='logits')
        vgg.fc3_W = W
        vgg.fc3_B = B

    with tf.name_scope('loss'):
        vgg.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.logits,
                                                                                   labels=labels_placeholder, name='xentropy_loss'))
        vgg.opt_loss = vgg.loss
        if args.l2reg>1e-20:
            vgg.opt_loss += args.l2reg * tf.reduce_sum(W*W)
        if args.l1reg>1e-20:
            vgg.opt_loss += args.l1reg * tf.reduce_sum(tf.abs(W))
    

def add_optimizer(vgg, args):
    vgg.global_step = tf.Variable(0, trainable=False)
    if args.lr_decay_rate > 0 or args.lr_decay_steps > 0 or args.staircase:
        assert args.lr_decay_rate > 0, "one of lr_decay_rate, lr_decay_steps, staircaes is set, so you must set lr_decay_rate > 0"
        assert args.lr_decay_steps > 0, "one of lr_decay_rate, lr_decay_steps, staircaes is set, so you must set lr_decay_steps > 0"
        vgg.learning_rate = tf.train.exponential_decay(args.lr, vgg.global_step, args.lr_decay_steps, args.lr_decay_rate, staircase=args.staircase)
    else:
        vgg.learning_rate = args.lr

    if args.opt.lower() == 'momentum':
        vgg.optimizer = tf.train.MomentumOptimizer(learning_rate=vgg.learning_rate,
                                                   momentum=args.mom)
    elif args.opt.lower() in ['gradientdescent', 'graddescent', 'grad']:
        vgg.optimizer = tf.train.GradientDescentOptimizer(learning_rate=vgg.learning_rate)
    else:
        raise Exception("opt not understood: %s" % args.opt)
    
def run_init_get_sess_saver(args, saver_dict=None):
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads = args.intra_op_parallelism_threads))    
    init = tf.initialize_all_variables()
    if saver_dict is None:
        saver = tf.train.Saver()
    else:
        saver = tf.train.Saver(saver_dict)
    sess.run(init)
    return sess, saver

def make_model(args, datareader):
    num_outputs = datareader.num_outputs()
    img_orig = tf.placeholder(tf.float32, shape=datareader.features_placeholder_shape())
    img_vgg16 = preprocess.imgbatch_2_vgg16(imgs=img_orig, channel_mean=8.46)
    labels_placeholder = tf.placeholder(tf.float32, shape=(None, num_outputs))
    train_placeholder = tf.placeholder(tf.bool)
    vgg = vgg16(imgs=img_vgg16, weights=None, sess=None, trainable=args.trainable, stop_at_fc2=True)
    add_loss(vgg, labels_placeholder, num_outputs, args)
    add_optimizer(vgg, args)
    train_ops = [vgg.optimizer.minimize(vgg.loss, global_step=vgg.global_step)]
    predict_op = tf.nn.softmax(vgg.logits)
    sess, saver = run_init_get_sess_saver(args, saver_dict={'fc3_W':vgg.fc3_W,
                                                            'vc3_B':vgg.fc3_B})    
    return psmlearn.Model(X_placeholder         = img_orig,
                          Y_placeholder         = labels_placeholder,
                          trainflag_placeholder = train_placeholder,
                          X_processed           = img_vgg16,
                          nnet                  = vgg,
                          train_ops             = train_ops,
                          predict_op            = predict_op,
                          sess                  = sess,
                          saver                 = saver)

def train(args, datareader, model):
    if model is None:
        model = make_model(args, datareader)
        if not args.dev: model.nnet.load_weights(weight_file='data/vgg16_weights.npz', sess=model.sess)
    validation_imgs_orig, validation_labels = datareader.get_validation_set()
    validation_imgs_vgg16 = model.sess.run(model.X_processed, {model.X_placeholder: validation_imgs_orig})
    validation_feed_dict = {model.X_processed:validation_imgs_vgg16,
                            model.Y_placeholder:validation_labels,
                            model.trainflag_placeholder:False}    


    tfTrainer = psmlearn.TensorFlowTrainer(args,
                                           validation_feed_dict=validation_feed_dict,
                                           model=model,
                                           datareader=datareader,
                                           print_cmat=True)
    tfTrainer.train()

    return model

def predict(args, datareader, model):
    if model is None:
        model = make_model(args, datareader)
        if not args.dev: model.nnet.load_weights(weight_file='data/vgg16_weights.npz', sess=model.sess)
        model.saver.restore(model.sess, save_path=args.train_save)
        
    batches = 10
    cmat = None
    for step in range(batches):
        X, Y = datareader.get_next_minibatch()
        feed_dict = {model.X_placeholder:X,
                     model.trainflag_placeholder:False}
        predict = model.sess.run(model.predict_op, feed_dict=feed_dict)
        cur_cmat = psmlearn.util.get_confusion_matrix_one_hot(predict, Y)
        if cmat is None:
            cmat = cur_cmat
        else:
            cmat += cur_cmat
        print("batch %d of %d: acc=%.2f" % (step, batches, np.trace(cur_cmat)/np.sum(cur_cmat)))
    acc, cmatrows = psmlearn.util.cmat2str(cmat)
    msg = "acc/cmat on %d batches of training data: =%.2f" % (batches, acc)
    msg += '   '
    N = len(msg)
    print("%s%s" % (msg, cmatrows.pop(0)))
    for row in cmatrows:
        print("%s%s" % (' '*N, row))
    return model

def plot_images(plt, img, bprop, gbprop, label, score, pause=0.2):
    img = np.maximum(0.0, img.astype(np.float))
    img /= np.max(img)
    log_img = np.log(1+1000.0*img)
    log_img /= np.max(log_img)

    gbprop = gbprop.astype(np.float) - np.min(gbprop)
    gbprop = np.maximum(0.0, gbprop)
    gbprop /= np.max(gbprop)

    bprop = bprop.astype(np.float) - np.min(bprop)
    bprop = np.maximum(0.0, bprop)
    bprop /= np.max(bprop)


    plt.subplot(2,2,1)
    plt.imshow(img, interpolation='none')
    plt.title('vgg16 input')

    plt.subplot(2,2,2)
    plt.imshow(log_img, interpolation='none')
    plt.title('log vgg16 input')

    plt.subplot(2,2,3)
    plt.imshow(bprop, interpolation='none')
    plt.title('bprop lbl=%d score=%.2f' % (label, score))

    plt.subplot(2,2,4)
    plt.imshow(gbprop, interpolation='none')
    plt.title('gbprop')

    if pause is not None and pause > 0:
        plt.pause(pause)
    else:
        raw_input("hit enter")
    
def gbprop(args, datareader, model):
    import matplotlib as mpl
    mpl.rcParams['backend'] = 'TkAgg'
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(figsize=(22,16))
    plt.show()

    if model is None:
        model = make_model(args, datareader)
        if not args.dev: model.nnet.load_weights(weight_file='data/vgg16_weights.npz', sess=model.sess)
        model.saver.restore(model.sess, save_path=args.train_save)

    saliencyMap = psmlearn.SaliencyMap(model)
    
    batches = 10
    for step in range(batches):
        X, Y = datareader.get_next_minibatch()
        feed_dict = {model.X_placeholder:X,
                     model.trainflag_placeholder:False}
        predict = model.sess.run(model.predict_op, feed_dict=feed_dict)
        for label in [1,2,3]:
            row, score = psmlearn.util.get_best_correct_one_hot(predict, Y, label)
            if row is None:
                print("nothing correct for label=%d in batch=%d" % (label, step))
                continue
            assert np.abs(predict[row,label]-score)<1e-4
            assert np.argmax(predict[row,:])==label
            raw_img = X[row:(row+1),:,:,:]
            assert len(raw_img.shape)==4
            proc_img, bprop_img = saliencyMap.calc(raw_img, label, fn='bprop')
            proc_img, gbprop_img = saliencyMap.calc(raw_img, label, fn='gbprop')
            plot_images(plt, proc_img, bprop_img, gbprop_img, label, score, pause=None)
    return model
