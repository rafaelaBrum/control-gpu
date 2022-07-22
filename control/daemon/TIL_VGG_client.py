#!/usr/bin/env python3
# -*- coding: utf-8

import os
import argparse
import numpy as np
import random
import flwr as fl

# Scikit learn
from sklearn import metrics

from pandas import DataFrame

from time import time

# Filter warnings
import warnings

warnings.filterwarnings('ignore')

# Keras imports
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical

# Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    v1 = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    v1.logging.set_verbosity(v1.logging.ERROR)

# Project imports
from CacheManager_daemon import CacheManager
from CellRep_daemon import CellRep
from vgg_daemon import VGG16

# Supported image types
img_types = ['svs', 'dicom', 'nii', 'tif', 'tiff', 'png']


def print_confusion_matrix(y_pred, expected, classes, label):
    stp = len(expected)
    # x is expected, y is actual
    m_conf = np.zeros((classes + 3, classes + 1))
    for i in range(stp):
        m_conf[expected[i]][y_pred[i]] = m_conf[expected[i]][y_pred[i]] + 1
    m_conf_2 = m_conf.tolist()
    # Total predictions and expectations for each class
    for i in range(classes):
        m_conf_2[classes][i] = "{0:.0f}".format(
            sum(m_conf.transpose()[i]))
        m_conf_2[i][classes] = "{0:.0f}".format(sum(m_conf[i]))
    # Correct rate
    for i in range(classes):
        m_conf_2[classes + 1][i] = "{0:.0f}/{1:.0f}".format(
            m_conf[i][i], sum(m_conf.transpose()[i]))
    # Accuracy
    for i in range(classes):
        m_conf_2[classes + 2][i] = "{0:.4f}".format(
            m_conf[i][i] / sum(m_conf.transpose()[i]))

    # Copying m_conf2 to m_conf
    for i in range(classes):
        m_conf[classes][i] = m_conf_2[classes][i]
        m_conf[i][classes] = m_conf_2[i][classes]
        m_conf[classes + 2][i] = m_conf_2[classes + 2][i]

    # Total samples
    m_conf_2[classes][classes] = "{0:.0f}".format(m_conf.sum())
    m_conf_2[classes + 1][classes] = ''
    m_conf_2[classes + 2][classes] = '{0:.4f}'.format(sum(np.diag(m_conf)) / m_conf.sum())
    # Store accuracy in m_conf also
    m_conf[classes + 2][classes] = m_conf_2[classes + 2][classes]

    col = [i for i in range(classes)] + ['Expected Total']
    ind = [i for i in range(classes)] + \
          ['Predicted Total', 'Correct Rate', 'Accuracy']

    df = DataFrame(m_conf_2, columns=col, index=ind)
    print("Confusion matrix ({0}):".format(label))
    print(df)
    print('\n')

    # fd = open(
    #     os.path.join(os.path.abspath(args.logdir), 'confusion_matrix_{0}-nn{1}.csv'.format(label, args.network)),
    #     'w')
    # df.to_csv(fd, columns=col, index=ind, header=True)
    # fd.close()

    return m_conf


def get_args():
    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Flower client of Convolunional Neural '
                                                 'Network for Image Segmentation.')

    # #Preprocessing options
    pre_args = parser.add_argument_group('Preprocessing', 'Includes image format usage, tiling and normalization')
    arg_groups.append(pre_args)

    pre_args.add_argument('-predst', dest='predst', type=str, default='tiles',
                          help='Output tiles go to this directory')
    pre_args.add_argument('-tdim', dest='tdim', nargs='+', type=int,
                          help='Tile width and heigth, optionally inform the number of channels '
                               '(Use: 200 200 for SVS 50 um).',
                          default=None, metavar=('Width', 'Height'))

    # #Training options
    train_args = parser.add_argument_group('Training', 'Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False,
                            help='Train model')
    train_args.add_argument('-net', dest='network', type=str, default='VGG16',
                            help='Network name which should be trained.\n Check documentation for available models.')
    train_args.add_argument('-data', dest='data', type=str, help='Dataset name to train model.\n '
                                                                 'Check documentation for available datasets.',
                            default='CellRep')
    train_args.add_argument('-b', dest='batch_size', type=int,
                            help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float,
                            help='Learning rate (Default: 0.00005).', default=0.00005)
    train_args.add_argument('-tn', action='store_true', dest='new_net',
                            help='Do not use older weights file.', default=False)
    train_args.add_argument('-nsw', action='store_false', dest='save_w',
                            help='Do not save intermediate weights as a callback.', default=True)
    train_args.add_argument('-tnorm', action='store_true', dest='batch_norm',
                            help='Applies batch normalization during training.', default=False)
    train_args.add_argument('-aug', action='store_true', dest='augment',
                            help='Applies data augmentation during training.', default=False)
    train_args.add_argument('-plw', action='store_true', dest='plw',
                            help='Preload Imagenet weights after single model build.', default=False)
    train_args.add_argument('-lyf', dest='lyf', type=int,
                            help='Freeze this number of layers for training (Default=0).', default=0)
    train_args.add_argument('-wpath', dest='weights_path',
                            help='Use weights file contained in path - usefull for sequential training '
                                 '(Default: None).',
                            default='ModelWeights')
    train_args.add_argument('-split', dest='split', nargs=3, type=float,
                            help='Split data in as much as 3 sets (Default: 80%% train, 10%% validation, 10%% test). '
                                 'If AL experiment, test set can be defined as integer.',
                            default=(0.8, 0.1, 0.1), metavar=('Train', 'Validation', 'Test'))
    train_args.add_argument('-f1', dest='f1period', type=int,
                            help='Execute F1 and ROC AUC calculations every X epochs (Default: 0).', default=0)
    train_args.add_argument('-sample', dest='sample', type=float,
                            help='Use a sample of the whole data for training '
                                 '(Default: 100.0%% - use floats [0.0-1.0]).',
                            default=1.0)
    train_args.add_argument('-pos_rt', dest='pos_rt', type=float,
                            help='When sampling, keep a positive rate among samples '
                                 '(Default: not set - use floats [0.0-1.0]).',
                            default=None)

    # Model selection
    model_args = parser.add_argument_group('Model')
    arg_groups.append(model_args)

    model_args.add_argument('-model_dir', dest='model_path',
                            help='Save trained models in dir (Default: TrainedModels).',
                            default='TrainedModels')

    # Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int,
                         help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int,
                         help='Number of CPU cores available (Default: 1).', default=1)

    # Runtime options
    parser.add_argument('-out', dest='bdir', type=str, default='',
                        help='Base dir to store all temporary data and general output', required=True)
    parser.add_argument('-cache', dest='cache', type=str, default='cache',
                        help='Keeps caches in this directory', required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False,
                        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str, default='logs',
                        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False,
                        help='Keep loaded images in memory.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=False,
                        help='Delay the loading of images to the latest moment possible (memory efficiency).')

    parser.add_argument('-pred_size', dest='pred_size', type=int,
                        help='Limit test set size to this number of images.', default=0)
    parser.add_argument('-test_dir', dest='testdir', type=str, default=None,
                        help='Runs prediction on a different set of images stored in dir.')
    parser.add_argument('-wsi_split', dest='wsi_split', type=int,
                        help='Randomly select this number of WSIs to choose test patches from '
                             '(Default 0, means use all).',
                        default=0)
    parser.add_argument('-wsilist', dest='wsilist', type=str, nargs='+', default=None)

    # Flower args
    flwr_args = parser.add_argument_group('Flower args')
    arg_groups.append(flwr_args)

    flwr_args.add_argument("-server_address", dest='server_address', type=str, required=True,
                           help=f"gRPC server address", default='localhost:8080')
    flwr_args.add_argument("-epochs", type=int, required=True, default=10,
                           help="Number of epochs per round of federated learning")

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def _reduce_lr_on_epoch(epoch, lr):
    # Reduces LR by a factor of 10 every 30 epochs
    if epoch > 9 and not (epoch % 30):
        lr /= 10
    return lr


class Trainer(object):
    """
    Class that implements the training procedures applicable to all
    CNN models.

    Specialized training my be needed for some models and those should be
    implemented elsewhere.

    @param args <argparse config>: configurations as specified by user
    @param ds <datasource>: some subclass of a GenericDatasource
    """

    def __init__(self, args):
        """
        @param args <parsed configurations>: configurations
        """
        self._args = args
        self._verbose = args.verbose
        self._ds = None
        self._ds_test = None
        self._rex = r'{0}-t(?P<try>[0-9]+)e(?P<epoch>[0-9]+).h5'

    def load_modules(self):
        self._ds = CellRep(self._args.predst, self._args.keepimg, self._args)

        self._ds_test = CellRep(self._args.testdir, self._args.keepimg, self._args)

        net_model = VGG16(self._args, self._ds)

        return net_model

    def start_execution(self):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.
        """

        net_model = self.load_modules()

        # Test set splitting done in the same code now, outside GenericDatasource
        self.x_test, self.y_test, X, Y = split_test(self._args, self._ds)
        # self.x_test, self.y_test, _, _ = split_test(self._args, self._ds_test, test=True)

        self._rex = self._rex.format(net_model.name)

        # Define training data
        if self._args.sample != 1.0:
            X, Y, self.sample_idx = self._ds.sample_metadata(self._args.sample, data=(X, Y),
                                                             pos_rt=self._args.pos_rt)
            self._ds.check_paths(X, self._args.predst)

        # After test set is separated, after data sampling is done, now split train and val
        self.train_data, self.val_data = self._ds.split_metadata(self._args.split[:2], data=(X, Y))

        self.build_model(net_model)

    def _choose_generator(self, train_data, val_data, fix_dim):
        """B
        Returns a tuple with two batch generators: (train_generator, val_generator)
        The type of generator depends on the config.delay_load option
        """
        train_generator, val_generator, test_generator = (None, None, None)

        # if self._args.delay_load:
        #     X, Y = self.x_test, self.y_test
        # else:
        #     X, Y = self._ds.load_data(data=(self.x_test, self.y_test), keep_img=self._args.keepimg)

        if self._args.augment:
            train_prep = ImageDataGenerator(
                samplewise_center=self._args.batch_norm,
                samplewise_std_normalization=self._args.batch_norm,
                rotation_range=180,
                width_shift_range=20,
                height_shift_range=20,
                zoom_range=.2,
                # shear_range=.05,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=(-20.0, 20.0))

            val_prep = ImageDataGenerator(
                samplewise_center=self._args.batch_norm,
                samplewise_std_normalization=self._args.batch_norm,
                brightness_range=(-20.0, 20.0))
        else:
            train_prep = ImageDataGenerator(
                samplewise_center=self._args.batch_norm,
                samplewise_std_normalization=self._args.batch_norm)
            val_prep = ImageDataGenerator(
                samplewise_center=self._args.batch_norm,
                samplewise_std_normalization=self._args.batch_norm)

        test_prep = ImageDataGenerator(samplewise_center=self._args.batch_norm,
                                       samplewise_std_normalization=self._args.batch_norm)

        if self._args.delay_load:
            from BatchGenerator_daemon import ThreadedGenerator

            train_generator = ThreadedGenerator(dps=train_data,
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._args.batch_size,
                                                image_generator=train_prep,
                                                extra_aug=self._args.augment,
                                                shuffle=True,
                                                verbose=self._verbose,
                                                keep=self._args.keepimg)

            val_generator = ThreadedGenerator(dps=val_data,
                                              classes=self._ds.nclasses,
                                              dim=fix_dim,
                                              batch_size=self._args.batch_size,
                                              image_generator=val_prep,
                                              extra_aug=self._args.augment,
                                              shuffle=True,
                                              verbose=self._verbose,
                                              keep=self._args.keepimg)

            test_generator = ThreadedGenerator(dps=(self.x_test, self.y_test),
                                               classes=self._ds.nclasses,
                                               dim=fix_dim,
                                               batch_size=2*self._args.batch_size,
                                               image_generator=test_prep,
                                               extra_aug=self._args.augment,
                                               shuffle=False,
                                               verbose=self._verbose,
                                               input_n=1,
                                               keep=self._args.keepimg)
        else:
            # Loads training images and validation images
            x_train, y_train = self._ds.load_data(split=None, keep_img=self._args.keepimg, data=train_data)

            x_val, y_val = self._ds.load_data(split=None, keep_img=self._args.keepimg, data=val_data)

            # Labels should be converted to categorical representation
            y_train = to_categorical(y_train, self._ds.nclasses)
            y_val = to_categorical(y_val, self._ds.nclasses)
            train_generator = train_prep.flow(x_train, y_train, batch_size=self._args.batch_size, shuffle=True)
            val_generator = val_prep.flow(x_val, y_val, batch_size=1)

            X, Y = self._ds.load_data(data=(self.x_test, self.y_test), keep_img=self._args.keepimg)

            Y = to_categorical(Y, self._ds.nclasses)
            test_generator = test_prep.flow(x=X, y=Y, batch_size=2*self._args.batch_size, shuffle=False)
            del Y

        return train_generator, val_generator, test_generator

    def build_model(self, model, **kwargs):
        """
        Generic trainer. Receives a GenericModel and trains it
        @param model <GenericModel>
        @param train_data <list>
        @param val_data <list>

        Optional keyword arguments:
        @param set_session <boolean>: configure session here
        @param verbose <int>: set verbosity level for training process. If not specified, use default
        @param summary <boolean>: print model summary
        @param clear_sess <boolean>: clears session and frees GPU memory
        @param allocated_gpus <int>: currently not used
        @param save_numpy <boolean>: save weights in numpy format instead of HDF5
        """
        # @param train_data <list>: Should be a collection of image metadata
        # @param val_data <list>: Should be a collection of image metadata

        train_data = self.train_data
        val_data = self.val_data

        if 'set_session' in kwargs:
            set_session = kwargs['set_session']
        else:
            set_session = True

        if 'stats' in kwargs:
            stats = kwargs['stats']
        else:
            stats = None

        if 'summary' in kwargs:
            summary = kwargs['summary']
        else:
            summary = True

        if 'allocated_gpus' in kwargs:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = self._args.gpu_count

        # session setup
        if set_session:
            session = K.get_session()
            ses_config = tf.ConfigProto(
                device_count={"CPU": self._args.cpu_count, "GPU": self._args.gpu_count},
                intra_op_parallelism_threads=self._args.cpu_count if self._args.gpu_count == 0
                else self._args.gpu_count,
                inter_op_parallelism_threads=self._args.cpu_count if self._args.gpu_count == 0
                else self._args.gpu_count,
                log_device_placement=True if self._verbose > 1 else False
            )
            session.config = ses_config
            K.set_session(session)

        if self._verbose > 0 and (stats is None or stats):
            unique, count = np.unique(train_data[1], return_counts=True)
            l_count = dict(zip(unique, count))
            if len(unique) > 2:
                print("Training items:")
                print("\n".join(["label {0}: {1} items".format(key, l_count[key]) for key in unique]))
            else:
                if unique.shape[0] == 1:
                    l_count[unique[0] ^ 1] = 0
                print("Train labels: {0} are 0; {1} are 1;\n - "
                      "{2:.2f} are positives".format(l_count[0], l_count[1], (l_count[1] / (l_count[0] + l_count[1]))))

            unique, count = np.unique(val_data[1], return_counts=True)
            l_count = dict(zip(unique, count))
            if len(unique) > 2:
                print("Validation items:")
                print("\n".join(["label {0}: {1} items".format(key, l_count[key]) for key in unique]))
            else:
                if 1 not in l_count:
                    l_count[1] = 0
                print("Validation labels: {0} are 0; {1} are 1;\n - "
                      "{2:.2f} are positives".format(l_count[0], l_count[1], (l_count[1] / (l_count[0] + l_count[1]))))

            print("Train set: {0} items".format(len(train_data[0])))
            print("Validate set: {0} items".format(len(val_data[0])))

        self.train_generator, self.val_generator, self.test_generator = self._choose_generator(
            train_data, val_data, model.check_input_shape())

        single, parallel = model.build(data_size=len(train_data[0]),
                                       allocated_gpus=allocated_gpus,
                                       preload_w=self._args.plw,
                                       layer_freeze=self._args.lyf)
        if parallel is not None:
            self.training_model = parallel
        else:
            self.training_model = single

        if self._args.info and summary:
            print(single.summary())
            print("Model parameters: {}".format(single.count_params()))
            print("Model layers: {}".format(len(single.layers)))

        # self.pred_model, _ = model.build(training=False, pre_load_w=False)

    def train(self, epochs):

        old_e_offset = 0
        wf_header = "{0}-t{1}".format('VGG16', old_e_offset + 1)

        # Define special behaviour CALLBACKS
        callbacks = []
        # ModelCheckpoint
        if self._args.save_w:
            callbacks.append(ModelCheckpoint(os.path.join(
                self._args.weights_path, wf_header + "e{epoch:02d}.h5"),
                save_weights_only=True, period=5, save_best_only=True, monitor='val_acc'))
        # ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.7,
                                           patience=10, verbose=self._verbose,
                                           mode='auto', min_lr=1e-7))
        # LearningRateScheduler
        callbacks.append(LearningRateScheduler(_reduce_lr_on_epoch, verbose=self._verbose))

        hist = self.training_model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=len(self.train_generator),  # self._args.batch_size,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),  # self._config.batch_size,
            epochs=epochs,
            verbose=self._verbose,
            use_multiprocessing=False,
            workers=self._args.cpu_count*2,
            max_queue_size=self._args.batch_size*3,
            callbacks=callbacks,
            )

        if self._verbose > 1:
            print("Done training model: {0}".format(hex(id(self.training_model))))

        return hist

    def evaluate(self):

        old_e_offset = 0
        wf_header = "{0}-t{1}".format('VGG16', old_e_offset + 1)

        hist = self.training_model.evaluate_generator(
            generator=self.test_generator,
            steps=len(self.test_generator),  # self._args.batch_size,
            # epochs=epochs,
            verbose=self._verbose,
            use_multiprocessing=False,
            workers=self._args.cpu_count * 2,
            max_queue_size=self._args.batch_size * 3,
        )

        Y_pred = self.training_model.predict_generator(
            generator=self.test_generator,
            steps=len(self.test_generator),  # self._args.batch_size,
            # epochs=epochs,
            verbose=self._verbose,
            use_multiprocessing=False,
            workers=self._args.cpu_count*2,
            max_queue_size=self._args.batch_size*3,
            )

        y_pred = np.argmax(Y_pred, axis=1)
        expected = np.array(self.y_test)
        nclasses = self._ds.nclasses

        print("expected ({1}):\n{0}".format(expected, expected.shape))
        print("Predicted ({1}):\n{0}".format(y_pred, y_pred.shape))

        f1 = metrics.f1_score(expected, y_pred, pos_label=1)
        print("F1 score: {0:.2f}".format(f1))

        m_conf = print_confusion_matrix(y_pred, expected, nclasses, "TILs")

        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        neg_accuracy = 0.0
        pos_accuracy = 0.0


        # ROC AUC
        # Get positive scores (binary only)
        if nclasses == 2:
            scores = Y_pred.transpose()[1]
            fpr, tpr, thresholds = metrics.roc_curve(expected, scores, pos_label=1)
            print("AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))

            print("Accuracy: {0:.3f}".format(m_conf[nclasses + 2][nclasses]))

            print("m_conf", m_conf)

            neg_accuracy = m_conf[4][0]
            pos_accuracy = m_conf[4][1]
            precision = m_conf[1][1] / m_conf[2][1]
            recall = m_conf[1][1] / m_conf[1][2]
            f1_score = 2 * m_conf[1][1] / (m_conf[1][2] + m_conf[2][1])

            # print("False positive rates: {0}".format(fpr))
            # print("True positive rates: {0}".format(tpr))
            # print("Thresholds: {0}".format(thresholds))

        # if self._verbose > 1:
        print("Done evaluate model: {0}".format(hex(id(self.training_model))))

        return hist, neg_accuracy, pos_accuracy, precision, recall, f1_score

    def get_model_weights(self):
        return self.training_model.get_weights()

    def set_model_weights(self, parameters):
        # self.pred_model.set_weights(parameters)
        return self.training_model.set_weights(parameters)

    def get_train_data_length(self):
        return len(self.train_data[0])

    def get_val_data_length(self):
        return len(self.val_data[0])

    def get_test_data_length(self):
        return len(self.x_test)

    def get_epochs(self):
        return self._args.epochs


def main_exec(args):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(args.bdir):
        os.mkdir(args.bdir)

    if not os.path.isdir(args.weights_path):
        os.mkdir(args.weights_path)

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    if args.train:
        if not os.path.isdir(args.weights_path):
            os.mkdir(args.weights_path)
        if not os.path.isdir(args.model_path):
            os.mkdir(args.model_path)

        time_start = time()

        trainer = Trainer(args)
        trainer.start_execution()

        time_end = time()

        print("TF configuration time:", str(time_end-time_start))

        # Start Flower client
        client = VGG16Client(trainer)
        fl.client.start_numpy_client(args.server_address, client=client)

    if not args.train:
        print("The problem begins with choice: preprocess, train, postprocess or predict")


def _split_origins(args, x_data, t_idx):
    """
    Separates patches of a predefined number of WSIs to be used as test set
    """

    cache_m = CacheManager()
    if cache_m.check_file_existence('testset.pik'):
        full_id, samples = cache_m.load('testset.pik')
        if samples is not None and args.info:
            print("[DataSetup] Using cached TEST SET. This is DANGEROUS. Use the metadata correspondent to the set.")
        return full_id, samples

    wsis = set()

    for k in x_data:
        wsis.add(k.getOrigin())

    # Defines slides to provide test set patches
    wsis = list(wsis)
    if args.wsilist is None:
        selected = set(random.choices(wsis, k=args.wsi_split))
    else:
        selected = set(args.wsilist)
    selected_idx = []

    if args.info:
        print("[DataSetup] WSIs selected to provide test patches:\n{}".format("\n".join(selected)))

    patch_count = {}

    for i in range(len(x_data)):
        w = x_data[i].getOrigin()
        if w in selected:
            patch_count.setdefault(w, [])
            patch_count[w].append(i)

    if args.wsimax is None or len(args.wsimax) != len(args.wsilist):
        for w in patch_count:
            if args.info:
                print("[Datasetup] Using all {} patches from slide {}".format(len(patch_count[w]), w))
            selected_idx.extend(patch_count[w])
    else:
        for i in range(len(args.wsilist)):
            w = args.wsilist[i]
            pc = int(args.wsimax[i] * len(patch_count[w]))
            pc = min(pc, len(patch_count[w]))
            selected_idx.extend(patch_count[w][:pc])
            if args.info:
                print("[Datasetup] Using {} ({:.2f}%) patches from slide {}".format(pc, 100 * pc / len(patch_count[w]),
                                                                                    w))

    t_idx = min(len(selected_idx), t_idx)
    samples = np.random.choice(selected_idx, t_idx, replace=False)
    full_id = np.asarray(selected_idx, dtype=np.int32)
    cache_m.dump((full_id, samples), 'testset.pik')

    return full_id, samples


def split_test(args, ds, test=False):
    # Test set is extracted from the last items of the full DS or from a test dir and is not changed for the whole run
    fX, fY = ds.load_metadata()
    test_x = None
    test_y = None

    X = None
    Y = None

    if test:
        test_x = fX
        test_y = fY
    else:
        tsp = args.split[-1:][0]
        t_idx = 0
        if tsp > 1.0:
            t_idx = int(tsp)
        elif tsp > 0.0:
            t_idx = int(tsp * len(fX))
        else:
            t_idx = np.inf

        # Configuration option that limits test set size
        t_idx = min(args.pred_size, t_idx) if args.pred_size > 0 else t_idx

        if args.testdir is None or not os.path.isdir(args.testdir):
            if args.wsi_split > 0 or args.wsilist is not None:
                full_id, samples = _split_origins(args, fX, t_idx)
                test_x = fX[samples]
                test_y = fY[samples]
                X = np.delete(fX, full_id)
                Y = np.delete(fY, full_id)
            else:
                test_x = fX[- t_idx:]
                test_y = fY[- t_idx:]
                X, Y = fX[:-t_idx], fY[:-t_idx]
            ds.check_paths(test_x, args.predst)
        else:
            x_test, y_test = ds.run_dir(args.testdir)
            t_idx = min(len(x_test), t_idx)
            samples = np.random.choice(len(x_test), t_idx, replace=False)
            test_x = [x_test[s] for s in samples]
            test_y = [y_test[s] for s in samples]
            del x_test
            del y_test
            del samples
            X, Y = fX, fY

    return test_x, test_y, X, Y


# Flower client
class VGG16Client(fl.client.NumPyClient):
    def __init__(self, model:Trainer):
        self.model = model

    def get_parameters(self):
        # print("tipo weights", type(self.model.get_model_weights()))
        # weights = self.model.get_model_weights()
        # import pickle
        #
        # # store list in binary file so 'wb' mode
        # with open('weights.bin', 'wb') as fp:
        #     pickle.dump(weights, fp)
        #     print('Done writing list into a binary file')
        return self.model.get_model_weights()

    def fit(self, parameters, config):
        timeout = int(config["timeout"]) if "timeout" in config else None
        # partial_updates = bool(int(config["partial_updates"]))

        time_start = time()

        self.model.set_model_weights(parameters)

        # Get hyperparameters for this round
        epochs = self.model.get_epochs()

        history = self.model.train(epochs)

        # Return updated model parameters and results
        parameters_prime = self.model.get_model_weights()
        num_examples_train = self.model.get_train_data_length()

        time_end = time()

        fit_duration = time_end - time_start
        if timeout is not None:
            if fit_duration > timeout:
                parameters_prime = []

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "num_examples_ceil": num_examples_train*epochs,
            "num_examples": num_examples_train*epochs,
            "fit_duration": fit_duration
        }
        print("num_examples fit:", num_examples_train)
        print("fit results: ", results)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_model_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        history, neg_acc, pos_acc, precision, recall, f1_score  = self.model.evaluate()
        num_examples_test = self.model.get_test_data_length()
        loss = history[0]
        accuracy = history[1]
        print("num_examples fit:", num_examples_test)
        print("evaluate loss: ", loss)
        print("evaluate accuracy: ", accuracy)
        return loss, num_examples_test, {"accuracy": accuracy, "neg_accuracy": neg_acc, "pos_accuracy": pos_acc,
                                         "precision": precision, "recall": recall, "f1_score": f1_score}


if __name__ == "__main__":
    args, unparsed = get_args()

    files = {
        'metadata.pik': os.path.join(args.cache, '{0}-metadata.pik'.format(args.data)),
        'sampled_metadata.pik': os.path.join(args.cache, '{0}-sampled_metadata.pik'.format(args.data)),
        'testset.pik': os.path.join(args.cache, '{0}-testset.pik'.format(args.data)),
        'split_ratio.pik': os.path.join(args.cache, '{0}-split_ratio.pik'.format(args.data)),
        'data_dims.pik': os.path.join(args.cache, '{0}-data_dims.pik'.format(args.data))}

    cache_m = CacheManager(locations=files)

    args.split = tuple(args.split)
    # Run main program
    main_exec(args)
