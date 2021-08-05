#!/usr/bin/env python3
# -*- coding: utf-8

import os
from abc import ABC, abstractmethod
import numpy as np

# Filter warnings
import warnings

warnings.filterwarnings('ignore')

# Preparing migration to TF 2.0
import tensorflow as tf

if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    # tf.disable_v2_behavior()

# Keras imports
from keras.layers import Average, Concatenate
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras import regularizers

# Locals
from CacheManager_daemon import CacheManager


def load_model_weights(config, genmodel, tmodel, sw_thread=None):
    """
    genmodel: GenericModel
    tmodel: tuple (single_model,parallel_model) or a Keras.Model instance
    """
    import time
    from datetime import timedelta

    npfile = False
    checkpath = None
    stime = None

    if sw_thread is not None:
        last_thread = None
        if isinstance(sw_thread, list):
            last_thread = sw_thread[-1]
        else:
            last_thread = sw_thread
        if config.ffeat is None and last_thread.is_alive():
            if config.info:
                print("[load_model_weights] Waiting for model weights to become available...")
            last_thread.join()

    if hasattr(genmodel, 'get_npweights_cache'):
        checkpath = genmodel.get_npweights_cache(add_ext=True)
        spath = checkpath
        npfile = True

    if npfile and not os.path.isfile(checkpath):
        spath = genmodel.get_weights_cache()
        npfile = False

    if config.info:
        stime = time.time()

    # Model can be loaded from previous acquisition train or from a fixed final model
    if config.gpu_count > 1:
        if hasattr(genmodel, 'get_npmgpu_weights_cache'):
            checkpath = genmodel.get_npmgpu_weights_cache(add_ext=True)
            ppath = checkpath
            npfile = True

        if npfile and not os.path.isfile(checkpath):
            ppath = genmodel.get_mgpu_weights_cache()
            npfile = False

        if isinstance(tmodel, tuple):
            pred_model = tmodel[1]
        else:
            pred_model = tmodel
        if config.ffeat is not None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat, by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(ppath, allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
        elif os.path.isfile(ppath):
            pred_model.load_weights(ppath, by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
        else:
            if config.info:
                print("No trained model or weights file found")
            pred_model = None
    else:
        if isinstance(tmodel, tuple):
            pred_model = tmodel[0]
        else:
            pred_model = tmodel
        if config.ffeat is not None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat, by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(spath, allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))
        elif os.path.isfile(spath):
            pred_model.load_weights(spath, by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))
        else:
            if config.info:
                print("No trained model or weights file found")
            pred_model = None

    if config.info:
        etime = time.time()
        td = timedelta(seconds=(etime - stime))
        print("Weights loading took: {}".format(td))

    return pred_model


class GenericEnsemble(ABC):
    """
    Defines common model attributes and methods

    Some models may implement a feature extractor or an ensemble build. If they do, the following methods
    will be available:
    - build_extractor
    - build_ensemble
    """

    def __init__(self, config, ds, name=None):
        self._config = config
        self._ds = ds
        self.name = name
        self.single = None
        self.parallel = None
        self.tmodels = None
        self.tmids = None

    def reset(self):
        if hasattr(self, '_s_ensemble'):
            del self._s_ensemble
        if hasattr(self, '_p_ensemble'):
            del self._p_ensemble

        self.tmodels = None
        self.tmids = None

    def get_npweights_cache(self, add_ext=False):
        """
        Returns path to model cache.

        @param add_ext <boolean>
        """
        # @param add_ext <boolean>: add numpy file extension to file name.
        if add_ext:
            return "{}.npy".format(self.cache_m.file_location(self._weightsCache).split('.')[0])
        else:
            return self.cache_m.file_location(self._weightsCache).split('.')[0]

    def get_npmgpu_weights_cache(self, add_ext=False):
        """
        Returns path to model cache

        @param add_ext <boolean>
        """
        # @param add_ext <boolean>: add numpy file extension to file name.
        if add_ext:
            return "{}.npy".format(self.cache_m.file_location(self._mgpu_weightsCache).split('.')[0])
        else:
            return self.cache_m.file_location(self._mgpu_weightsCache).split('.')[0]

    # def adjustWeightCache(self):
    def adjust_weight_cache(self):

        self.cache_m.register_file(os.path.join(self._config.weights_path, self._weightsCache), self._weightsCache)
        self.cache_m.register_file(os.path.join(self._config.weights_path, self._mgpu_weightsCache),
                                   self._mgpu_weightsCache)
        self.cache_m.register_file(os.path.join(self._config.model_path, self._modelCache), self._modelCache)

    # def setName(self, name):
    def set_name(self, name):
        """
        Override GenericModel's method to redefine cache file names
        """
        self.name = name
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self._modelCache = "{0}-model.h5".format(self.name)

        self.adjust_weight_cache()

    def register_ensemble(self, m):
        self._model_n = m
        self._weightsCache = "{0}-EM{1}-weights.h5".format(self.name, m)
        self._mgpu_weightsCache = "{0}-EM{1}-mgpu-weights.h5".format(self.name, m)
        self._modelCache = "{0}-EM{1}-model.h5".format(self.name, m)

        self.adjust_weight_cache()

    def return_model_n(self):
        if hasattr(self, '_model_n'):
            return self._model_n
        else:
            return -1

    def build_extractor(self, **kwargs):
        """
        Builds a feature extractor from an already built model.

        Returns: Keras backend function

        Key word arguments:
        model: constructs the extractor from this model, build and ensemble if a dict of models is given
        preload_w: return model with weights already loaded? True -> Yes
        parallel: return parallel model (overrides gpu_count avaliation)? True -> Yes
        """
        # Weight loading for the feature extraction is done latter by requesting party
        model = None
        parallel = kwargs.get('parallel', False)

        if 'model' not in kwargs:
            print("[GenericEnsemble] A compiled model should be passed as the model argument")
            return None
        else:
            model = kwargs['model']

        if isinstance(model, dict):
            if self._config.info:
                print("Building ensemble from trained models: {}".format([hex(id(em)) for em in model.values()]))
            kwargs['emodels'] = model
            s_model, p_model = self.build_ensemble(**kwargs)
            if parallel and p_model is not None:
                model = p_model
            else:
                model = s_model

        f = None
        if self.is_ensemble():
            if parallel:
                p_features = [model.get_layer('EM{}-{}'.format(e, self.name)) for e in range(self._config.emodels)]
                layers = [imodel.get_layer('feature').output for imodel in p_features]
            else:
                layers = [model.get_layer('EM{}-{}'.format(e, 'feature')).output for e in range(self._config.emodels)]
            x = Concatenate()(layers)
            f = K.function(model.inputs, [x])
        else:
            layer = model.get_layer('feature')
            f = K.function(model.inputs, [layer.output])

        return f

    def build_ensemble(self, **kwargs):
        """
        Builds an ensemble of M Inception models.

        Weights are loaded here when new is set True because of the way ensembles should be built.

        Default build: avareges the output of the corresponding softmaxes

        @param feature <boolean>: ensemble should be a feature extractor (no classification top)
        @param npfile <boolean>: loads weights from numpy files
        @param rbuild <boolean>: build a new ensemble body or use the last built
        @param new <boolean>: create a new ensemble from emodels or use one already available
        @param emodels <dict>: dictionary  keys (int - model number) -> ensemble tower
        @param sw_thread <list/thread object>: wait on thread to load weights
        @param allocated_gpus <int>: define the number of GPUs to distribute the model
        @param load_weights <boolean>: load ensemble model weights during build
        """

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        # Optional parameters
        npfile = kwargs.get('npfile', False)
        rbuild = kwargs.get('rbuild', False)
        new = kwargs.get('new', False)
        load_weights = kwargs.get('load_weights', False)
        emodels = kwargs.get('emodels', None)
        sw_thread = kwargs.get('sw_thread', None)
        allocated_gpus = kwargs.get('allocated_gpus', self._config.gpu_count)
        feature = False

        inputs = None
        s_models = None
        p_models = None

        # Use cached models if needed
        if emodels is not None:
            ids = tuple([id(emodels[m]) for m in range(len(emodels))])
            if ids != self.tmids:
                self.tmids = ids
                self.tmodels = emodels
            else:
                new = False

        if emodels is None and self.tmodels is not None:
            emodels = self.tmodels

        if rbuild or (emodels is None and not (hasattr(self, '_s_ensemble') or hasattr(self, '_p_ensemble'))):
            if self._config.info and not new:
                print("[{}] No previous ensemble models stored, building new ones".format(self.name))
            s_models, p_models, inputs = self._build_ensemble_body(feature, npfile, allocated_gpus, sw_thread)
            p_models = list(filter(lambda x: not x is None, p_models))
        elif emodels is not None and (new or not (hasattr(self, '_s_ensemble') and hasattr(self, '_p_ensemble'))):
            # Build from trained models
            inputs = []
            outputs = []
            for e in emodels:
                if load_weights:
                    self.register_ensemble(e)
                    emodels[e] = load_model_weights(self._config, self, emodels[e], sw_thread)
                for l in emodels[e].layers:
                    l.name = 'EM{}-{}'.format(e, l.name)
                inputs.extend(emodels[e].inputs)
                outputs.append(emodels[e].layers[-1].output)
            if allocated_gpus > 1:
                p_models = list(emodels.values())
                p_outputs = outputs
            else:
                s_models = list(emodels.values())
                s_outputs = outputs
        else:
            return self._s_ensemble, self._p_ensemble

        s_model, p_model = None, None
        if s_models is not None and len(s_models) > 0:
            s_outputs = [out for s in s_models for out in s.outputs]
            x = Average()(s_outputs)
            s_model = Model(inputs=inputs, outputs=x)

        if p_models is not None and len(p_models) > 0:
            p_outputs = [out for p in p_models for out in p.outputs]
            x = Average()(p_outputs)
            p_model = Model(inputs=inputs, outputs=x)

        if hasattr(self, '_s_ensemble'):
            del self._s_ensemble
        if hasattr(self, '_p_ensemble'):
            del self._p_ensemble

        self._s_ensemble = s_model
        self._p_ensemble = p_model

        return s_model, p_model

    def _build_ensemble_body(self, feature, npfile, allocated_gpus, sw_thread=None):
        s_models = []
        p_models = []
        inputs = []

        width, height, channels = self.check_input_shape()
        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        for m in range(self._config.emodels):
            self.register_ensemble(m)
            model, inp = self._build_architecture(input_shape=input_shape, training=False,
                                                  feature=feature, preload=False,
                                                  ensemble=True)

            inputs.append(inp)

            # Updates all layer names to avoid repeated name error
            for layer in model.layers:
                layer.name = 'EM{}-{}'.format(m, layer.name)

            single, parallel = self._configure_compile(model, allocated_gpus)

            single, parallel = self._load_weights(single, parallel, npfile, m, sw_thread)

            s_models.append(single)
            p_models.append(parallel)

        return s_models, p_models, inputs

    def _load_weights(self, single, parallel, npfile, m='', sw_thread=None):

        if sw_thread is not None:
            last_thread = None
            if isinstance(sw_thread, list):
                last_thread = sw_thread[-1]
            else:
                last_thread = sw_thread
            if last_thread.is_alive():
                if self._config.info:
                    print("[GenericEnsemble] Waiting for model weights to become available...")
                last_thread.join()

        if parallel is not None:
            # Updates all layer names to avoid repeated name error
            for layer in parallel.layers:
                layer.name = 'EM{}-{}'.format(m, layer.name)
            if npfile:
                parallel.set_weights(np.load(self.get_npmgpu_weights_cache(add_ext=True), allow_pickle=True))
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name,
                                                                    self.get_npmgpu_weights_cache(add_ext=True)
                                                                    ))
            elif os.path.isfile(self.get_mgpu_weights_cache()):
                parallel.load_weights(self.get_mgpu_weights_cache(), by_name=True)
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name, self.get_mgpu_weights_cache()))

            return single, parallel

        if npfile:
            single.set_weights(np.load(self.get_npweights_cache(add_ext=True), allow_pickle=True))
            if self._config.info:
                print("[{}] loaded ensemble weights: {}".format(self.name, self.get_npweights_cache(add_ext=True)))
        elif os.path.isfile(self.get_weights_cache()):
            single.load_weights(self.get_weights_cache(), by_name=True)
        else:
            if self._config.info:
                print("[{}] Could not load ensemble weights (model {})".format(self.name, m))
            single = None

        return single, parallel

    # def rescaleEnabled(self):
    def rescale_enabled(self):
        """
        Returns if the network is rescalable
        """
        return False

    def check_input_shape(self):
        # Image shape by OpenCV reports height x width
        if self._config.tdim is not None:
            if len(self._config.tdim) == 2:
                dims = [(None,) + tuple(self._config.tdim) + (3,)]
            else:
                dims = [(None,) + tuple(self._config.tdim)]
        elif self._ds is not None:
            dims = self._ds.get_dataset_dimensions()
        else:
            dims = [(None, 100, 100, 3)]

        # Dataset may have images of different sizes. What to do? Currently, chooses the smallest....
        _, width, height, channels = dims[0]

        return width, height, channels

    @abstractmethod
    def get_model_cache(self):
        pass

    @abstractmethod
    def get_weights_cache(self):
        pass

    @abstractmethod
    def get_mgpu_weights_cache(self):
        pass

    @abstractmethod
    def _build(self, width, height, channels, **kwargs):
        pass

    def build(self, **kwargs):
        """
        Child classes should implement: _build method

        Optional params:
        @param data_size <int>: size of the training dataset
        @param training <boolean>: set layer behavior to training mode (aplicable to dropout/BatchNormalization)
        @param feature <boolean>: return features instead of softmax classification
        @param preload_w <boolean>: load pre-trained weights to model
        @param allocated_gpus <int>: number of GPU availables
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param keep_model <boolean>: store created model as an instance atribute.
        @param layer_freeze <int>: freeze this many layers for training
        """

        width, height, channels = self.check_input_shape()

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        training = kwargs.setdefault('training', None)
        feature = kwargs.setdefault('feature', False)
        preload = kwargs.setdefault('preload_w', False)
        keep_model = kwargs.setdefault('keep_model', True)
        layer_freeze = kwargs.setdefault('layer_freeze', 0)
        new = kwargs.setdefault('new', True)

        if not 'allocated_gpus' in kwargs or kwargs['allocated_gpus'] is None:
            kwargs['allocated_gpus'] = self._config.gpu_count

        if new or (self.single is None and self.parallel is None):
            model, parallel_model = self._build(width, height, channels, **kwargs)
            if keep_model:
                self.single = model
                self.parallel = parallel_model
        else:
            model, parallel_model = self.single, self.parallel

        return model, parallel_model

    def get_ds(self):
        return self._ds

    def is_ensemble(self):
        return self._config.strategy == 'EnsembleTrainer'

    # def getName(self):
    def get_name(self):
        return self.name


class KNet(GenericEnsemble):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: https://keras.io/examples/mnist_cnn/
    """
    def __init__(self, config, ds, name=None):
        super().__init__(config, ds, name=name)
        if name is None:
            self.name = "KerasNet"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
 
        self.cache_m = CacheManager()
        self.cache_m.register_file(os.path.join(config.model_path, self._modelCache), self._modelCache)
        self.cache_m.register_file(os.path.join(config.weights_path, self._weightsCache), self._weightsCache)
        self.cache_m.register_file(os.path.join(config.weights_path, self._mgpu_weightsCache), self._mgpu_weightsCache)

        self.single = None
        self.parallel = None
        
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

    def _build(self, width, height, channels, **kwargs):
        """
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param data_size <int>: size of the training dataset
        """
        training = kwargs.get('training', None)
        feature = kwargs.get('feature')
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape, training, feature)
        return self._configure_compile(model)

    def _configure_compile(self, model):
        # Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.register_file(os.path.join(self._config.cache, lr_cache), lr_cache)
        l_rate = 0.0005
        if os.path.isfile(self.cache_m.file_location(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        # opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        # opt = optimizers.Adam(lr = l_rate)
        opt = optimizers.Adadelta()

        # Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])

            parallel_model = multi_gpu_model(model, gpus=self._config.gpu_count)
            parallel_model.compile(loss='categorical_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'],
                                   # options=p_opt,
                                   # run_metadata=p_mtd
                                   )
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'],
                          # options=p_opt,
                          # run_metadata=p_mtd
                          )

        return model, parallel_model

    def _build_architecture(self, input_shape, training, feature):
            
        model = Sequential()
        model.add(Convolution2D(32, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=input_shape))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self._ds.nclasses, activation='softmax'))

        return model


class BayesKNet(KNet):
    """
    Bayesian model for the KNet
    """
    def __init__(self, config, ds):
        super(BayesKNet, self).__init__(config=config, ds=ds, name="BayesKNet")

    def build_extractor(self, **kwargs):
        """
        Builds a feature extractor
        """

        return self._build(**kwargs)
    
    def _build_architecture(self, input_shape, training, feature):
        if hasattr(self, 'data_size'):
            weight_decay = 2.5/float(self.data_size)
            if self._config.verbose > 1:
                print("Setting weight decay to: {0}".format(weight_decay))
        else:
            weight_decay = 0.01
            
        inp = Input(shape=input_shape)

        x = Convolution2D(32, (3, 3), input_shape=input_shape,
                          strides=1,
                          padding='valid',
                          name='block1_conv1')(inp)
        x = Activation('relu')(x)
        x = Convolution2D(32, (3, 3),
                          strides=1,
                          padding='valid',
                          name='block1_conv2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.25)(x, training=training)
        
        x = Flatten()(x)
        x = Dense(128, kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)        
        x = Dropout(0.5)(x, training=training)
        x = Dense(self._ds.nclasses)(x)
        output = Activation('softmax')(x)
        
        return Model(inp, output, name=self.name)
