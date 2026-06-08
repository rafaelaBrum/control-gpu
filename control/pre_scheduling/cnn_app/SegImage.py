#!/usr/bin/env python3
# -*- coding: utf-8

from abc import ABC, abstractmethod
import os


class SegImage(ABC):
    """
    Common abstract class for any image type supported by the system

    @param path <str>: path to image file
    @param keepImg <bool>: keep image data in memory until ordered to release it
    """
    def __init__(self, path, keep_img=False, verbose=0):
        
        self._path = path
        self._verbose = verbose
        self._keep = keep_img
        self._data = None
        self._dim = None

    def __str__(self):
        """
        String representation is file name
        """
        return os.path.basename(self._path)

    def __eq__(self, other):
        if not isinstance(other, SegImage):
            return False
        else:
            return self.get_img_name() == other.get_img_name()

    def __getstate__(self):
        """
        Prepares for pickling.
        """
        state = self.__dict__.copy()
        del state['_data']
        state['_data'] = None

        return state
      
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    # def readImage(self, size=None, verbose=None):
    def read_image(self, size=None, verbose=None):
        pass

    @abstractmethod
    # def readImageRegion(self, x, y, dx, dy):
    def read_image_region(self, x, y, dx, dy):
        pass
    
    @abstractmethod
    # def getImgDim(self):
    def get_img_dim(self):
        """
        Should return dimensions as a tuple of (widthd,height,channels)
        """
        pass
    
    # def setKeepImg(self, keep):
    def set_keep_img(self, keep):
        """
        If image should not be held anymore, delete data
        """
        if keep is None:
            return
        
        if not keep:
            del self._data
            self._data = None

        self._keep = keep
    
    # def getImgName(self):
    def get_img_name(self):
        return os.path.basename(self._path).split('.')[0]

    # def getPath(self):
    def get_path(self):
        return self._path

    # def setPath(self, new_path):
    def set_path(self, new_path):
        self._path = new_path
