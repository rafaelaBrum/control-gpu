#!/usr/bin/env python3
# -*- coding: utf-8

import os
import skimage
import skimage.transform
from skimage import io

from .SegImage import SegImage


class PImage(SegImage):
    """
    Represents any image handled by skimage.
    """
    def __init__(self, path, keep_img=False, origin=None, coord=None, verbose=0):
        """
        @param path <str>: path to image
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin
        @param coord <tuple>: coordinates in original image
        """
        super().__init__(path, keep_img, verbose)
        self._coord = coord
        self._origin = origin

    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord, self._origin)
        else:
            return os.path.basename(self._path)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Hashes current dir and file name
        return hash((self._path.split(os.path.sep)[-2], os.path.basename(self._path)))
    
    # def readImage(self, keepImg=None, size=None, verbose=None, toFloat=True):
    def read_image(self, keep_img=None, size=None, verbose=None, to_float=True):

        data = None

        if keep_img is None:
            keep_img = self._keep
        elif keep_img:
            # Change seting if we are going to keep the image in memory now
            self.set_keep_img(keep_img)
        if verbose is not None:
            self._verbose = verbose
            
        if self._data is None or size != self._dim:
            if self._verbose > 1:
                print("Reading image: {0}".format(self._path))
                
            data = io.imread(self._path)

            # Convert data to float and also normalizes between [0,1]
            if to_float:
                data = skimage.img_as_float32(data)
            
            if data.shape[2] > 3:  # remove the alpha
                data = data[:, :, 0:3]
                
            if size is not None and data.shape != size:
                if self._verbose > 1:
                    print("Resizing image {0} from {1} to {2}".format(os.path.basename(self._path), data.shape, size))
                data = skimage.transform.resize(data, size)
                
            h, w, c = data.shape
            self._dim = (w, h, c)
            
            if self._keep:
                self._data = data
                
        else:
            if self._verbose > 1:
                print("Data already loaded:\n - {0}".format(self._path))
            data = self._data

        return data
    
    # def readImageRegion(self, x, y, dx, dy):
    def read_image_region(self, x, y, dx, dy):
        data = None
        
        if self._data is None:
            data = self.read_image()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    # def getImgDim(self):
    def get_img_dim(self):
        """
        Implements abstract method of SegImage
        """
        h, w, c = 0, 0, 0

        if self._dim is not None:
            return self._dim
        elif self._data is not None:
            h, w, c = self._data.shape
        else:
            data = io.imread(self._path)
            if data.shape[2] > 3:  # remove the alpha
                data = data[:, :, 0:3]
            h, w, c = data.shape
            
            if self._keep:
                self._data = data

        self._dim = (w, h, c)
        return self._dim

    # def getOrigin(self):
    def get_origin(self):
        return self._origin

    # def getCoord(self):
    def get_coord(self):
        if self._coord is not None and self._coord[0].isdigit() and self._coord[1].isdigit():
            return int(self._coord[0]), int(self._coord[1])
        else:
            if self._verbose > 1:
                print("[PImage] Image has incompatible coordinates: {}".format(self._coord))
            return None
