#!/usr/bin/env python3
# -*- coding: utf-8

import os
import argparse
import json

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Project imports
# from Preprocessing import Preprocess
# from Testing import TrainTest, DatasourcesTest,PredictionTest,ActiveLearningTest
from cnn_app.CacheManager import CacheManager
from cnn_app.GenericTrainer import TrainerSingletonFactory
from time import time


# Supported image types
img_types = ['svs', 'dicom', 'nii', 'tif', 'tiff', 'png']


def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(config.bdir):
        os.mkdir(config.bdir)

    if not os.path.isdir(config.weights_path):
        os.mkdir(config.weights_path)
        
    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.logdir):
        os.mkdir(config.logdir)

    if not os.path.isdir(config.weights_path):
        os.mkdir(config.weights_path)
    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    times_epochs = {}

    if config.info:
        print("Starting training process....")

    trainer = TrainerSingletonFactory.get_instance(config)
    trainer.start_execution()

    time_start = time()

    trainer.train(net_name=config.network, epochs=config.epochs)

    time_end = time()

    times_epochs['fit_1'] = str(time_end-time_start)

    time_start = time()

    trainer.evaluate()

    time_end = time()

    times_epochs['eval_1'] = str(time_end - time_start)

    time_start = time()

    trainer.train(net_name=config.network, epochs=config.epochs)

    time_end = time()

    times_epochs['fit_2'] = str(time_end - time_start)

    time_start = time()

    trainer.evaluate()

    time_end = time()

    times_epochs['eval_2'] = str(time_end - time_start)

    print("times_epochs")
    print(times_epochs)

    with open(config.file, 'w') as f:
        f.write(json.dumps(times_epochs))


if __name__ == "__main__":

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolutional Neural Network for Image Segmentation.')

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

    train_args.add_argument('-net', dest='network', type=str, default='',
                            help='Network name which should be trained.\n Check documentation for available models.')
    train_args.add_argument('-data', dest='data', type=str,
                            help='Dataset name to train model.\n '
                                 'Check documentation for available datasets.', default='')
    train_args.add_argument('-b', dest='batch_size', type=int,
                            help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float,
                            help='Learning rate (Default: 0.00005).', default=0.00005)
    train_args.add_argument('-e', dest='epochs', type=int,
                            help='Number of epochs (Default: 1).', default=1)
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
    parser.add_argument('-i', action='store_true', dest='info', default=False,
                        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-logdir', dest='logdir', type=str, default='logs',
                        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False,
                        help='Keep loaded images in memory.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=False,
                        help='Delay the loading of images to the latest moment possible (memory efficiency).')

    # #Run prediction options
    parser.add_argument('-pred_size', dest='pred_size', type=int,
                        help='Limit test set size to this number of images.', default=0)
    parser.add_argument('-test_dir', dest='testdir', type=str, default=None,
                        help='Runs prediction on a different set of images stored in dir.')
    parser.add_argument('-wsi_split', dest='wsi_split', type=int,
                        help='Randomly select this number of WSIs to choose test patches from '
                             '(Default 0, means use all).', default=0)
    parser.add_argument('-wsilist', dest='wsilist', type=str, nargs='+', default=None)
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False,
                        help='Print progress bars of processing execution.')

    # Pre Scheduling options
    parser.add_argument('-file', dest='file', type=str, default='times.json', help='File to print execution times')

    config, unparsed = parser.parse_known_args()
    
    files = {
        'datatree.pik': os.path.join(config.cache, '{}-datatree.pik'.format(config.data)),
        'tcga.pik': os.path.join(config.cache, 'tcga.pik'),
        'metadata.pik': os.path.join(config.cache, '{0}-metadata.pik'.format(config.data)),
        'sampled_metadata.pik': os.path.join(config.cache, '{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik': os.path.join(config.cache, '{0}-testset.pik'.format(config.data)),
        'initial_train.pik': os.path.join(config.cache, '{0}-inittrain.pik'.format(config.data)),
        'split_ratio.pik': os.path.join(config.cache, '{0}-split_ratio.pik'.format(config.data)),
        'clusters.pik': os.path.join(config.cache, '{0}-clusters.pik'.format(config.data)),
        'data_dims.pik': os.path.join(config.cache, '{0}-data_dims.pik'.format(config.data)),
        'tiles.pik': os.path.join(config.predst, 'tiles.pik'),
        'test_pred.pik': os.path.join(config.logdir, 'test_pred.pik'),
        'cae_model.h5': os.path.join(config.model_path, 'cae_model.h5'),
        'vgg16_weights_notop.h5': os.path.join('PretrainedModels', 'vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)

    config.split = tuple(config.split)
    # Run main program
    main_exec(config)
