#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Network
from keras.models import Sequential,Model
from keras.layers import Input,Activation
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras.applications import vgg16
from keras import regularizers
from keras_contrib.layers import GroupNormalization
from keras import backend as K

#Locals
from .CacheManager import CacheManager
from .GenericEnsemble import GenericEnsemble

class VGG16(GenericEnsemble):
    """
    Implements abstract methods from GenericModel.
    Producess a VGG16 model as implemented by Keras, with convolutional layers
    FC layers are substituted by Conv2D, as defined in:
    https://github.com/ALSM-PhD/quip_classification/blob/master/NNFramework_TF/sa_networks/vgg.py
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "VGG16_A1"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self.cache_m = CacheManager()
        self.cache_m.register_file(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.register_file(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.register_file(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)

    def get_model_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.file_location(self._modelCache)
    
    def get_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.file_location(self._weightsCache)
    
    def get_mgpu_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.file_location(self._mgpu_weightsCache)
    
    def _build(self,width,height,channels,**kwargs):
        """
        Returns a VGG 16 model instance, final fully-connected layers are substituted by Conv2Ds
        
        @param pre_trained <boolean>: returned model should be pre-trained or not
        """
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']
            
        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape)
        
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.register_file(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = self._config.learn_r
        if os.path.isfile(self.cache_m.file_location(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))

        l_rate = self.rescale('lr',l_rate)
        sgd = optimizers.SGD(lr=l_rate, decay=1.0e-4, momentum=0.9, nesterov=True)
        #adam = optimizers.Adam(lr = l_rate)
        
        #Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

            parallel_model = multi_gpu_model(model,gpus=self._config.gpu_count)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=sgd,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        self.single = model
        self.parallel = parallel_model

        return (model,parallel_model)

    def _build_architecture(self,input_shape):
        original_vgg16 = vgg16.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_shape=input_shape)

        #Freeze initial layers, except for the last 3:
        #for layer in original_vgg16.layers[:-2]:
        #    layer.trainable = False
            
        model = Sequential()
        model.add(original_vgg16)
        model.add(Convolution2D(4096, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(4096, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('softmax'))

        return model
