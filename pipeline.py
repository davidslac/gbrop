from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import project_setup

import os
import sys
from glob import glob
import random
import argparse

def make_datareader(args):
    h5files = []
    runs = map(int,args.runs.split(','))
    for run in runs:
        h5files.extend(glob(os.path.join(args.datadir, 'amo86815_mlearn-r%3.3d*.h5' % run)))
    assert len(h5files)>0
    random.shuffle(h5files)
    if args.dev:
        h5files=h5files[0:5]
    datareader = H5MiniBatchReader(h5files=h5files,
                                   minibatch_size=args.minibatch_size,
                                   validation_size=args.valid_size,
                                   feature_dataset=args.X,
                                   label_dataset=args.Y,
                                   return_as_one_hot=True,
                                   feature_preprocess=None,
                                   number_of_batches=None,
                                   class_labels_max_imbalance_ratio=1.0,
                                   max_mb_to_preload_all=None,
                                   add_channel_to_2D='row_column_channel',
                                   random_seed=None,
                                   verbose=True)
    return datareader

def train(args, datareader, model):
    return vgg16net.train(args, datareader, model)

def predict(args, datareader, model):
    return vgg16net.predict(args, datareader, model)

def gbprop(args, datareader, model):
    return vgg16net.gbprop(args, datareader, model)

def pipeline(args):
    random.seed(args.seed)
    datareader = make_datareader(args)

    with tf.Graph().as_default():
        model = None
        if not (args.predict or args.gbprop):
            model = train(args, datareader, model)
        if not args.gbprop:
            model = predict(args, datareader, model)
        gbprop(args, datareader, model)


### MAIN #########################################################################

programDescription = \
'''pipeline for doing guided back propagation against xtcav using the vgg16 networks.
'''
    
programDescriptionEpilog = \
'''Steps are
1. train a top classification layer to predict 0,1,2,3 - the 4 lasing conditions.
2. run prediction to verify classifier works.
3. run gbprop to see results.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=programDescription,
                                     epilog=programDescriptionEpilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    default_seed = 92839
    default_l1reg = 0.01
    default_l2reg = 0.001
    default_optimizer = 'momentum'
    default_momentum = 0.9
    default_learning_rate = 0.2
    default_lr_decay_rate = 0.97
    default_lr_decay_steps = 50
    default_minibatch_size = 64
    default_validation_size = 128
    default_train_steps = 8000
    default_steps_between_evals = 50

    default_runs='70,71'
    default_datadir = '/reg/d/ana01/temp/davidsch/ImgMLearnSmall'
    default_X = 'xtcavimg'
    default_Y = 'acq.peaksLabel'
    default_dimg = 'gbprop'
    
    parser.add_argument('--dev', action='store_true', help='development mode')
    parser.add_argument('--seed', type=int, help='seed for python random module.', default=92839) 
    parser.add_argument('--l1reg', type=float, help='l1reg during training. def=%f' % default_l1reg, default=default_l1reg)
    parser.add_argument('--l2reg', type=float, help='l2reg during training. def=%f' % default_l2reg, default=default_l2reg)
    parser.add_argument('--opt', type=str, help='training optimizer, momentum, adadelta, adagrad, adam, ftrl or rmsprop. default=%s' % default_optimizer, default=default_optimizer)
    parser.add_argument('--mom', type=float, help='momentum optimizers momentum, default=%f' % default_momentum, default=default_momentum)    
    parser.add_argument('--lr', type=float, help='learning rate. defaults to %f' % default_learning_rate, default=default_learning_rate)
    parser.add_argument('--lr_decay_rate', type=float, help='decay rate for learning rate. defaults to %f' % default_lr_decay_rate, default=default_lr_decay_rate)
    parser.add_argument('--lr_decay_steps', type=int, help='decay steps for learning rate. defaults to %d' % default_lr_decay_steps, default=default_lr_decay_steps)
    parser.add_argument('--staircase', action='store_true', help='staricase for learning rate decay')
    parser.add_argument('--trainable', action='store_true', help='make all vgg variables trainable')
    parser.add_argument('--train_steps', type=int, help='number of training steps default=%d' % default_train_steps, default=default_train_steps)
    parser.add_argument('--train_save', type=str, help='name of trainer to save', default='vgg16_t12_model')
    parser.add_argument('--force', action='store_true', help='force overwrite of existing model name')
    parser.add_argument('--eval_steps', type=int, help='number of steps between evals default=%d' % default_steps_between_evals, default=default_steps_between_evals)
    parser.add_argument('--minibatch_size', type=int, help='minibatch size, default=%d' % default_minibatch_size, default=default_minibatch_size)
    parser.add_argument('--valid_size', type=int, help='validation size, default=%d' % default_validation_size, default=default_validation_size)
    parser.add_argument('--intra_op_parallelism_threads', type=int, help='number of intra op threads, default=12', default=12)

    parser.add_argument('--runs',     type=str, help='comma separated list of runs to process for dataset, default=%s' % default_runs, default=default_runs)
    parser.add_argument('--datadir',  type=str, help='full path to data directory default=%s' % default_datadir, default=default_datadir)
    parser.add_argument('--X',        type=str, help='dataset for features, default=%s' % default_X, default=default_X)
    parser.add_argument('--Y',        type=str, help='dataset for labels/Y, default=%s' % default_Y, default=default_Y)
    parser.add_argument('--predict',  action='store_true', help='jump to prediction')
    parser.add_argument('--gbprop',   action='store_true', help='jump to gbprop')
    parser.add_argument('--dimg',     type=str, help='saliency map, one of bprop gpropb, default=%s' % default_dimg, default=default_dimg)

    args = parser.parse_args()
    import tensorflow as tf
    from h5minibatch.H5MiniBatchReader import H5MiniBatchReader
    import vgg16net

    pipeline(args)
    
