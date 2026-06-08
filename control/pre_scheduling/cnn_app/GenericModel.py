#!/usr/bin/env python3
# -*- coding: utf-8

from abc import ABC, abstractmethod
import math
import numpy as np


class GenericModel(ABC):
    """
    Defines common model attributes and methods

    Some models may implement a feature extractor or an ensemble build. If they do, the following methods
    will be available:
    - build_extractor
    - build_ensemble
    """
    def __init__(self, config, ds, name):
        self._config = config
        self._ds = ds
        self.name = name
        self.single = None
        self.parallel = None

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
    def _build(self, **kwargs):
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

    # def setName(self, name):
    def set_name(self, name):
        self.name = name

    # def getName(self):
    def get_name(self):
        return self.name

    def rescale(self, dim, full_size):
        """
        Rescales down a network according to inversed EfficientNet strategy.
        - [Efficientnet: Rethinking model scaling for convolutional neural networks ] (https://arxiv.org/pdf/1905.11946)

        Params:
        - dim <string>: 'depth', 'width', 'resolution' or 'lr';
        - phi <int>: rescaling factor (should be compatible with network architecture). Equivalent to reducing resources
        by 1/phi;
        - full_size <int,list,tuple>: original size which will be scaled down.

        Return: returns rescaled dimension of tuple if full_size is a list
        """
        alpha = 1.2
        beta = 1.1
        gama = 1.15
        phi = 1

        if phi <= 1:
            return full_size

        if not dim in ['depth', 'width', 'resolution', 'lr']:
            print("[GenericModel] Dim should be one of 'depth', 'width', 'resolution' or 'lr'")
            return full_size

        if dim == 'depth':
            rd = 1 / math.pow(alpha, phi)
        elif dim == 'width':
            rd = 1 / math.pow(beta, phi)
        elif dim == 'resolution':
            rd = 1 / math.pow(gama, phi)
        else:
            if phi > 2:
                rd = 2.5 * (1 + math.log(phi))
            else:
                rd = 2.5 * phi

        if isinstance(full_size, list) or isinstance(full_size, tuple):
            full_size = np.asarray(full_size, dtype=np.float32)
            full_size *= rd
            full_size.round(out=full_size)
            full_size = full_size.astype(np.int32)
            np.clip(full_size, a_min=1, a_max=full_size.max(), out=full_size)
            full_size = tuple(full_size)
        elif isinstance(full_size, int):
            full_size *= rd
            full_size = int(round(full_size))
            full_size = 1 if full_size < 1 else full_size
        else:
            full_size *= rd

        return full_size
