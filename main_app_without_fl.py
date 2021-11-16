#!/usr/bin/env python3
# -*- coding: utf-8

import os
import argparse

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Project imports
# from Preprocessing import Preprocess
# from Testing import TrainTest, DatasourcesTest,PredictionTest,ActiveLearningTest
from cnn_app.CacheManager import CacheManager
from cnn_app import GenericTrainer
from cnn_app import Predictions


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
        
    # if config.preprocess:
    #     if config.img_type is None:
    #         imgt = img_types
    #     else:
    #         imgt = config.img_type
    #
    #     Preprocess.preprocess_data(config,imgt)
        
    if config.train:
        if not os.path.isdir(config.weights_path):
            os.mkdir(config.weights_path)
        if not os.path.isdir(config.model_path):
            os.mkdir(config.model_path)
            
        GenericTrainer.run_training(config, None)

    if config.pred:
        Predictions.run_prediction(config,None)

    # if config.postproc:
    #     pass

    # if config.runtest:
    #     if config.tmode == 0:
    #         pass
    #     elif config.tmode == 1:
    #         #Run train test
    #         TrainTest.run(config)
    #     elif config.tmode == 2:
    #         DatasourcesTest.run(config)
    #     elif config.tmode == 3:
    #         PredictionTest.run(config)
    #     elif config.tmode == 4:
    #         ActiveLearningTest.run(config)

    # if not (config.preprocess or config.train or config.postproc or config.runtest):
    if not (config.train or config.pred):
        print("The problem begins with choice: preprocess, train, postprocess or predict")


if __name__ == "__main__":

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    # #Preprocessing options
    pre_args = parser.add_argument_group('Preprocessing', 'Includes image format usage, tiling and normalization')
    arg_groups.append(pre_args)
    
    pre_args.add_argument('--pre', action='store_true', dest='preprocess', default=False,
                          help='Run preprocess steps')
    pre_args.add_argument('-tile', action='store_true', dest='tile', default=False,
                          help='Make tiles from input images')
    pre_args.add_argument('-tcga', action='store_true', dest='tcga', default=False,
                          help='Input is a TCGA image base.')    
    pre_args.add_argument('-presrc', dest='presrc', type=str, default='',
                          help='Input image or directory of images (runs recursively)', required=False)
    pre_args.add_argument('-predst', dest='predst', type=str, default='tiles',
                          help='Output tiles go to this directory')
    pre_args.add_argument('-img_type', dest='img_type', nargs='+', type=str,
                          help='Input image types to consider (list): svs, dicom, nii.', default=None)
    pre_args.add_argument('-mag', dest='magnification', type=int,
                          help='For SVS images only, use specific magnification level.',
                          choices=[2, 4, 8, 10, 20, 40], default=40)
    pre_args.add_argument('-tdim', dest='tdim', nargs='+', type=int,
                          help='Tile width and heigth, optionally inform the number of channels '
                               '(Use: 200 200 for SVS 50 um).',
                          default=None, metavar=('Width', 'Height'))
    pre_args.add_argument('-norm', dest='normalize', type=str, nargs='?', default=None, 
                          const='Preprocessing/target_40X.png',
                          help='Normalize tiles based on reference image (given)')

    # #Training options
    train_args = parser.add_argument_group('Training', 'Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False,
                            help='Train model')
    train_args.add_argument('-net', dest='network', type=str, default='',
                            help='Network name which should be trained.\n Check documentation for available models.')
    train_args.add_argument('-data', dest='data', type=str, help='Dataset name to train model.\n '
                                                                 'Check documentation for available datasets.',
                            default='')
    train_args.add_argument('-b', dest='batch_size', type=int,
                            help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float,
                            help='Learning rate (Default: 0.00005).', default=0.01)
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
    train_args.add_argument('-f1', dest='f1period', type=int,
                            help='Execute F1 and ROC AUC calculations every X epochs (Default: 0).', default=0)
    train_args.add_argument('-met', dest='metricsperiod', type=int,
                            help='Calculate metrics in test dataset every X epochs (Default: 0).', default=0)
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

    # #Run prediction options
    parser.add_argument('--pred', action='store_true', dest='pred', default=False,
                        help='Runs prediction with a given model (use -net parameter).')
    parser.add_argument('-print', action='store_true', dest='print_pred', default=False,
                        help='Prints stored prediction results.')
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

    # #System tests
    test_args = parser.add_argument_group('Tests')
    arg_groups.append(test_args)
    
    parser.add_argument('-t', action='store_true', dest='runtest', default=False,
                        help='Run tests.')
    test_args.add_argument('-tmode', dest='tmode', type=int,
                           help='Run tests for individual subsystems: \n '
                                '0 - Run all tests; \n '
                                '1 - Run training test; \n '
                                '2 - Run Datasources test; \n '
                                '3 - Run Prediction test; \n '
                                '4 - Run AL test.',
                           choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('-tlocal', action='store_true', dest='local_test', default=False,
                        help='Test is local (assumes a small dataset).')
    
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
