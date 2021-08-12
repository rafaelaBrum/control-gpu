#!/usr/bin/env python3
# -*- coding: utf-8

import os

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

# Network
from keras.layers import Input
from keras import optimizers
from keras.utils import multi_gpu_model
from keras import backend as K
from keras import layers
from keras import utils as keras_utils
from keras import models
from keras_contrib.layers import GroupNormalization

# Locals
from CacheManager import CacheManager
from GenericEnsemble import GenericEnsemble


BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None,
              use_bn=True):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      kernel_initializer='glorot_normal',
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias and use_bn:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    else:
        bn_name = None if name is None else name + '_gn'
        x = GroupNormalization(groups=16, axis=-1, name=bn_name, scale=False)(x)

    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', use_bn=True):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(x, 32, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(branch_1, 32, 3, use_bn=use_bn)
        branch_2 = conv2d_bn(x, 32, 1, use_bn=use_bn)
        branch_2 = conv2d_bn(branch_2, 48, 3, use_bn=use_bn)
        branch_2 = conv2d_bn(branch_2, 64, 3, use_bn=use_bn)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(x, 128, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], use_bn=use_bn)
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], use_bn=use_bn)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(x, 192, 1, use_bn=use_bn)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], use_bn=use_bn)
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], use_bn=use_bn)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   use_bn=use_bn,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                      classes=1000, **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    use_bn = kwargs['batch_n']
    use_dp = kwargs.get('use_dp', True)
    name = kwargs.get('name', 'Inception')
    lf = kwargs.get('layer_freeze', 0)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', use_bn=use_bn)
    x = conv2d_bn(x, 32, 3, padding='valid', use_bn=use_bn)
    x = conv2d_bn(x, 64, 3, use_bn=use_bn)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid', use_bn=use_bn)
    x = conv2d_bn(x, 192, 3, padding='valid', use_bn=use_bn)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, use_bn=use_bn)
    branch_1 = conv2d_bn(x, 48, 1, use_bn=use_bn)
    branch_1 = conv2d_bn(branch_1, 64, 5, use_bn=use_bn)
    branch_2 = conv2d_bn(x, 64, 1, use_bn=use_bn)
    branch_2 = conv2d_bn(branch_2, 96, 3, use_bn=use_bn)
    branch_2 = conv2d_bn(branch_2, 96, 3, use_bn=use_bn)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, use_bn=use_bn)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx,
                                   use_bn=use_bn)

        if use_dp:
            x = layers.Dropout(0.1)(x, training=kwargs['training'])

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', use_bn=use_bn)
    branch_1 = conv2d_bn(x, 256, 1, use_bn=use_bn)
    branch_1 = conv2d_bn(branch_1, 256, 3, use_bn=use_bn)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', use_bn=use_bn)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx,
                                   use_bn=use_bn)

        if use_dp:
            x = layers.Dropout(0.2)(x, training=kwargs['training'])

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, use_bn=use_bn)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', use_bn=use_bn)
    branch_1 = conv2d_bn(x, 256, 1, use_bn=use_bn)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid', use_bn=use_bn)
    branch_2 = conv2d_bn(x, 256, 1, use_bn=use_bn)
    branch_2 = conv2d_bn(branch_2, 288, 3, use_bn=use_bn)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid', use_bn=use_bn)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx,
                                   use_bn=use_bn)
        if use_dp:
            x = layers.Dropout(0.15)(x, training=kwargs['training'])

    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10,
                               use_bn=use_bn)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b', use_bn=use_bn)

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='feature')(x)
        x = layers.Dense(classes, activation='softmax', kernel_initializer='glorot_normal', name='predictions')(x)
    elif 'custom_top' in kwargs and kwargs['custom_top']:
        # Create a custom new classification here if needed
        x = layers.GlobalAveragePooling2D(name='feature')(x)
        x = layers.Dense(512, kernel_initializer='glorot_normal')(x)
        x = layers.Activation('relu', name='class1_ac')(x)
        if use_dp:
            x = layers.Dropout(0.3)(x, training=kwargs['training'])
        x = layers.Dense(128, kernel_initializer='glorot_normal')(x)
        x = layers.Activation('relu', name='class2_ac')(x)
        if use_dp:
            x = layers.Dropout(0.3)(x, training=kwargs['training'])
        x = layers.Dense(classes, kernel_initializer='glorot_normal')(x)
        x = layers.Activation('softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='feature')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='feature')(x)
        x = layers.Dense(classes, activation='softmax', kernel_initializer='glorot_normal', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    with tf.device('/cpu:0'):
        model = models.Model(inputs, x, name=name)

    # Load weights.
    if 'preload' in kwargs and kwargs['preload']:
        if weights == 'imagenet':
            if include_top:
                fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = keras_utils.get_file(
                    fname,
                    BASE_WEIGHT_URL + fname,
                    cache_subdir='models',
                    file_hash='e693bd0210a403b3192acc6073ad2e96')
            else:
                fname = ('inception_resnet_v2_weights_'
                         'tf_dim_ordering_tf_kernels_notop.h5')
                weights_path = keras_utils.get_file(
                    fname,
                    BASE_WEIGHT_URL + fname,
                    cache_subdir='models',
                    file_hash='d19885ff4a710c122648d3b5c3b684e4')
            model.load_weights(weights_path, by_name=True)
        elif weights is not None:
            model.load_weights(weights, by_name=True)

        if lf > 0:
            count = 0
            convs = 0
            for l in model.layers:
                if isinstance(l, layers.Conv2D):
                    if count < lf:
                        l.trainable = False
                        count += 1
                    convs += 1
            print("Froze first {} conv layers (total convs:{})".format(count, convs))

    return model


class Inception(GenericEnsemble):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: 
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
    Addapted to provide a Bayesian model
    """
    def __init__(self, config, ds, name=None):
        super().__init__(config, ds, name=name)
        if name is None:
            self.name = "Inception"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
 
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path, self._modelCache), self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path, self._weightsCache), self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path, self._mgpu_weightsCache), self._mgpu_weightsCache)

        self.single = None
        self.parallel = None
        
    def get_model_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._modelCache)
    
    def get_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._weightsCache)

    def get_mgpu_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._mgpu_weightsCache)
    
    def _build(self, width, height, channels, **kwargs):
        """
        Custom build process
        """
        training = kwargs.get('training', None)
        feature = kwargs.get('feature')
        preload = kwargs.get('preload_w')
        lf = kwargs.get('layer_freeze')
        allocated_gpus = kwargs.get('allocated_gpus')
        
        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        model = self._build_architecture(input_shape, training, feature, preload, layer_freeze=lf)

        return self._configure_compile(model, allocated_gpus)

    def _configure_compile(self, model, allocated_gpus):
        """
        Configures, compiles, generates parallel model if needed

        @param model <Keras.Model>
        """
        # Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache, lr_cache), lr_cache)
        l_rate = self._config.learn_r
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        # opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        opt = optimizers.Adam(lr=l_rate)
        # opt = optimizers.Adadelta(lr=l_rate)

        # Return parallel model if multiple GPUs are available
        parallel_model = None
       
        if allocated_gpus > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])
            parallel_model = multi_gpu_model(model, gpus=allocated_gpus)
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

    def _build_architecture(self, input_shape, training=None, feature=False, preload=True, ensemble=False,
                            layer_freeze=0):

        """
        Parameters:
        - training <boolean>: sets network to training mode, wich enables dropout if there are DP layers
        - feature <boolean>: build a feature extractor - DEPRECATED
        - preload <boolean>: preload Imagenet weights
        - ensemble <boolean>: builds an ensemble of networks from the Inception architecture

        OBS: self.is_ensemble() returns if the ensemble strategy is in use
        """

        kwargs = {'training': training,
                  'feature': feature,
                  'custom_top': False,
                  'preload': preload,
                  'layer_freeze': layer_freeze,
                  'name': self.name,
                  'batch_n': True if self._config.gpu_count <= 1 else False,
                  'use_dp': True,  # False if self.is_ensemble() else True
                  'model': self}

        inp = Input(shape=input_shape)

        inception_body = InceptionResNetV2(include_top=False,
                                           weights='imagenet',
                                           input_tensor=inp,
                                           input_shape=input_shape,
                                           pooling='avg',
                                           classes=self._ds.nclasses,
                                           **kwargs)

        if ensemble:
            return inception_body, inp
        else:
            return inception_body
