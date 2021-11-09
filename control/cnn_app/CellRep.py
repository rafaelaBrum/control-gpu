#!/usr/bin/env python3
# -*- coding: utf-8

import os

# Local modules
from . import GenericDatasource as gd
from .PImage import PImage


class CellRep(gd.GenericDS):
    """
    Class that parses label.txt text files and loads all images into memory
    """

    def __init__(self, data_path, keep_img=False, config=None):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param config <argparse>: configuration object
        @param keepImg <boolean>: keep image data in memory
        """
        super().__init__(data_path, keep_img, config, name='CellRep')
        self.nclasses = 2

    def _load_metadata_from_dir(self, d):
        """
        Create SegImages from a directory
        """
        class_set = set()
        labels = open(os.path.join(d, 'label.txt'), 'r')

        t_x, t_y = ([], [])
        for f in labels:
            tmp = f.strip().split()
            f_name, f_label = tmp[0], tmp[1]
            origin = ''
            coord = None
            label = int(f_label)
            if label < 1:
                label = 0
            if len(tmp) > 2:
                origin = tmp[2]
            if len(tmp) > 4:
                coord = (tmp[3], tmp[4])
            t_path = os.path.join(d, f_name)
            if os.path.isfile(t_path):
                seg = PImage(t_path, keep_img=self._keep, origin=origin, coord=coord, verbose=self._verbose)
                t_x.append(seg)
                t_y.append(label)
                class_set.add(label)
            elif self._verbose > 1:
                print("Label file contains reference to {0}, but no such file exists.".format(t_path))

        # Non-lymphocyte patches are labeld 0 or -1 (no lymphocyte or below lymphocyte threshold)
        # -1 and 0 labels are treated as the same as for now this is a binary classification problem
        if self._verbose > 1:
            print("On directory {2}:\n - Number of classes: {0};\n - Classes: {1}".format(len(class_set), class_set,
                                                                                          os.path.basename(d)))

        return t_x, t_y

    def _release_data(self):
        del self.X
        del self.Y
        
        self.X = None
        self.Y = None

    def change_root(self, s, d):
        """
        s -> original path
        d -> change location to d
        """
        components = tuple(s.split(os.path.sep)[-2:])
        relative_path = os.path.join(*components)

        return os.path.join(d, relative_path)
