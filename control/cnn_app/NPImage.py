#!/usr/bin/env python3
# -*- coding: utf-8

import os
import numpy as np

from .SegImage import SegImage


class NPImage(SegImage):
    """
    Represents any image already stored as Numpy arrays.
    """

    def __init__(self, path, data=None, keep_img=False, origin=None, coord=None, verbose=0):
        """
        @param path <str>: path to image
        @param data <ndarray>: image data in a Numpy array
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin: x_train, x_val or x_test
        @param coord <int>: coordinates in original image: index
        """
        super().__init__(path, keep_img, verbose)
        self._coord = coord
        self._origin = origin
        if data is not None and isinstance(data, np.ndarray):
            self._data = data

    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord, self._origin)
        else:
            return "{0}-{1}".format(os.path.basename(self._path), self._coord)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Hashes current dir and file name
        return hash((os.path.basename(self._path), self._origin, self._coord))

    # def readImage(self, keepImg=None, size=None, verbose=None):
    def read_image(self, keep_img=None, size=None, verbose=None):

        if self._data is not None:
            return self._data

        data = None
        with np.load(self._path, allow_pickle=True) as f:
            if self._origin in f:
                data = f[self._origin][self._coord]

        if keep_img is None:
            keep_img = self._keep
        elif keep_img:
            # Change seting if we are going to keep the image in memory now
            self.set_keep_img(keep_img)
        if verbose is not None:
            self._verbose = verbose

        if self._keep:
            self._data = data

        return data

    # def getImgDim(self):
    def get_img_dim(self):
        """
        Implements abstract method of SegImage
        """

        if self._dim is not None:
            return self._dim
        elif self._data is not None:
            self._dim = self._data.shape
        else:
            data = self.read_image()
            self._dim = data.shape
                
        return self._dim

    # def readImageRegion(self, x, y, dx, dy):
    def read_image_region(self, x, y, dx, dy):
        data = None
        
        if self._data is None:
            data = self.read_image()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    def __getstate__(self):
        """
        Prepares for pickling.
        """
        state = self.__dict__.copy()
        if not self._keep:
            del state['_data']
            state['_data'] = None

        return state
