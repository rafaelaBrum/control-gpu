#!/usr/bin/env python3
# -*- coding: utf-8

import pickle
import os


class _CacheManager(object):
    """
    Manages cache files. Locations, loads, dumps.
    """
    __instance = None
    
    def __new__(cls, *args, **kwds):
        if _CacheManager.__instance is None:
            _CacheManager.__instance = object.__new__(_CacheManager)
            _CacheManager.__instance.__init__(*args, **kwds)
            return _CacheManager.__instance
        else:
            return _CacheManager.__instance

    def __init__(self, *args, **kwds):
        """
        Controls all __locations creation.

        @param locations <dict>: Use the locations specified in the given dictionary
        @param verbose <int>: verbosity level
        """
        if args:
            if isinstance(args[0], dict):
                self.__locations = {}
                self.__locations.update(args[0])
            if isinstance(args[1], int):
                self._verbose = args[1]
            else:
                self._verbose = 0
        if kwds:
            if 'locations' in kwds:
                self.__locations = {}
                self.__locations.update(kwds['locations'])
            if 'verbose' in kwds:
                self._verbose = kwds['verbose']
            else:
                self._verbose = 0

        self._multi_load_fd = None

    # def getLocations(self):
    def get_locations(self):
        return self.__locations.copy()
    
    # def fileLocation(self, fid):
    def file_location(self, fid):
        """
        Returns the registered file location (if it is registered)
        """

        if fid in self.__locations:
            return self.__locations[fid]
        else:
            return None

    # def checkFileExistence(self, fid):
    def check_file_existence(self, fid):
        """
        Returns True if cache file exists. False otherwise
        """
        if fid in self.__locations and os.path.isfile(self.__locations[fid]):
            return True
        else:
            return False
        
    def register_file(self, path, fid, overwrite=False):
        """
        Register the location where a file should be stored
        """
        if fid in self.__locations:
            if overwrite:
                self.__locations[fid] = path
            elif self._verbose > 0:
                print("[CacheManager] File ID already registered. Set overwrite to True if a new path should be stored")
        else:
            self.__locations[fid] = path
                
    def dump(self, data, fid, single=False):
        """
        Dumps data to file. If data is a list, dump each item one at a time.
        Uses pickle. Make sure data is pickable.

        @param single <boolean>
        @param data
        @param fid
        """
        # @param single <boolean>: dump whole data at once
        if fid in self.__locations:
            dump_dir = os.path.dirname(self.__locations[fid])
            if not os.path.isdir(dump_dir):
                os.makedirs(dump_dir)
            fd = open(self.__locations[fid], 'wb')
            if isinstance(data, list) and not single:
                for item in data:
                    pickle.dump(item, fd)
            else:
                pickle.dump(data, fd)
            fd.close()
        else:
            print("[CacheManager-DUMP] No such file ID registered: {0}".format(fid))

    def load(self, fid):
        """
        Loads data from file. Pickled files.
        """
        if fid in self.__locations and os.path.isfile(self.__locations[fid]):
            fd = open(self.__locations[fid], 'rb')
            data = pickle.load(fd)
            fd.close()
            return data
        else:
            if self._verbose > 0:
                print("[CacheManager-LOAD] No such file or ID not registered: {0}".format(fid))
            return None

    def load_file(self, f):
        if os.path.isfile(f):
            fd = open(f, 'rb')
            data = pickle.load(fd)
            fd.close()
            return data
        else:
            if self._verbose > 0:
                print("[CacheManager-LOAD] No such file or ID not registered: {0}".format(fid))
            return None
        
    def multi_load(self, fid):
        """
        Keeps file descriptor opened until end of file is reached. Use with attention.
        NOT THREAD SAFE!
        """
        if fid not in self.__locations or not os.path.isfile(self.__locations[fid]):
            return None
        
        if self._multi_load_fd is None:
            self._multi_load_fd = open(self.__locations[fid], 'rb')
            data = pickle.load(self._multi_load_fd)

            return data
        
        elif os.path.basename(self._multi_load_fd.name) == fid:
            data = pickle.load(self._multi_load_fd)
            
            # Check if EOF reached, close file, destroy fd
            if self._multi_load_fd.tell() == os.fstat(self._multi_load_fd.fileno()).st_size:
                self._multi_load_fd.close()
                self._multi_load_fd = None
                
            return data
            
        else:
            print("[CacheManager-MULTILOAD] File descriptor already opened for another file: {0}; "
                  "resquested: {1}".format(self._multi_load_fd.name, fid))
            return None

    def read(self, fid):
        """
        Reads from file as text.
        """
        data = None
        if fid in self.__locations:
            with open(self.__locations[fid], 'r') as fd:
                data = fd.read()

        return data


def CacheManager(*args, **kwds):
    cache_singleton = _CacheManager(*args, **kwds)

    return cache_singleton
