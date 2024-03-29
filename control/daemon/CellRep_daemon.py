#!/usr/bin/env python3
# -*- coding: utf-8

import os
import sys
import numpy as np
import multiprocessing

from tqdm import tqdm

import concurrent.futures
import random

import skimage
import skimage.transform
from skimage import io

# Local modules
from CacheManager_daemon import CacheManager


def multiprocess_run(exec_function, exec_params, data, cpu_count, step_size, output_dim=1, txt_label='',
                     verbose=False):
    """
    Runs exec_function in a process pool. This function should receive parameters as follows:
    (iterable_data,param2,param3,...), where paramN is inside exec_params

    @param exec_function <function>
    @param exec_params <tuple>
    @param data <iterable>
    @param cpu_count <int>
    @param step_size <int>
    @param output_dim <int>
    @param txt_label <str>
    @param verbose <bool>
    """

    # @param cpu_count <int>: use this number of cores
    # @param step_size <int>: size of the iterable that exec_function will receive
    # @param output_dim <int>: exec_function produces how many sets of results?

    # Perform extractions of frames in parallel and in steps
    step = int(len(data) / step_size) + (len(data) % step_size > 0)
    datapoints_db = [[] for i in range(output_dim)]
    semaphores = []

    process_counter = 0
    pool = multiprocessing.Pool(processes=cpu_count, maxtasksperchild=50,
                                initializer=tqdm.set_lock, initargs=(multiprocessing.RLock(),))

    datapoints = np.asarray(data)
    for i in range(step):
        # get a subset of datapoints
        end_idx = step_size

        if end_idx > len(data):
            end_idx = len(data)

        cur_datapoints = datapoints[:end_idx]

        semaphores.append(pool.apply_async(exec_function,
                                           args=(cur_datapoints,) + exec_params))

        datapoints = np.delete(datapoints, np.s_[:end_idx], axis=0)

        # datapoints = np.delete(datapoints,np.s_[i*step_size : end_idx],axis=0)
        # del cur_datapoints

    for i in range(len(semaphores)):
        res = semaphores[i].get()
        for k in range(output_dim):
            datapoints_db[k].extend(res[k])
        if verbose > 0:
            print("[{2}] Done transformations (step {0}/{1})".format(i, len(semaphores) - 1, txt_label))
            sys.stdout.flush()

    # Free all possible memory
    pool.close()
    pool.join()

    del datapoints

    # remove None points
    return tuple(filter(lambda x: not x is None, datapoints_db))


class CellRep:
    """
    Class that parses label.txt text files and loads all images into memory
    """

    def __init__(self, data_path, keep_img=False, config=None):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param config <argparse>: configuration object
        @param keepImg <boolean>: keep image data in memory
        """
        self.path = None
        if isinstance(data_path, str) and os.path.isdir(data_path):
            self.path = data_path
        else:
            raise ValueError("[GenericDatasource] Path does not correspond to a directory ({0}).".format(data_path))

        self.X = None
        self.Y = None
        self.name = 'CellRep'
        self.multi_dir = True
        self._cache = CacheManager()
        self._keep = keep_img
        self._cpu_count = config.cpu_count if config is not None else 1
        self._verbose = config.verbose if config is not None else 0
        self._config = config
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

    def check_paths(self, imgv, path):

        for s in imgv:
            s.set_path(self.change_root(s.get_path(), path))

    def get_dataset_dimensions(self, ds=None):
        """
        Returns the dimensions of the images in the dataset. It's possible to have different image dimensions.
        WARNING: big datasets will take forever to run. For now, checks a sample of the images.
        TODO: Reimplement this function to be fully parallel (threads in case).

        Return: SORTED list of tuples (# samples,width,height,channels)
        """

        cache_m = CacheManager()
        reload_data = False
        if cache_m.check_file_existence('data_dims.pik'):
            try:
                dims, name = cache_m.load('data_dims.pik')
            except ValueError:
                reload_data = True
            if name != self.name:
                reload_data = True
        else:
            reload_data = True

        if reload_data:
            dims = set()
            if ds is None and self.X is None:
                return None
            elif ds is None:
                ds = self.X

            samples = len(ds)
            if self._config.info:
                print("Checking a sample of dataset images for different dimensions...")

            s_number = int(0.02 * samples)
            upper_limit = 5000 if s_number > 5000 else s_number
            for seg in random.sample(ds, upper_limit):
                dims.add((samples,) + seg.get_img_dim())
            cache_m.dump((dims, self.name), 'data_dims.pik')

        l = list(dims)
        l.sort()
        return l

    def _run_multiprocess(self, data):
        """
        This method should not be called directly. It's intended
        only for multiprocess metadata loading.
        """
        X, Y = ([], [])
        for item in data:
            t_x, t_y = self._load_metadata_from_dir(item)
            X.extend(t_x)
            Y.extend(t_y)

        return X, Y

    def _split_data(self, split, X, Y, test_size):
        """
        Split data in at most N sets. Returns a tuple (set1,set2,set3,setN) with the divided
        data
        """
        if sum(split) <= 1.0:
            it_count = 0
            split_data = []
            start_idx = 0
            samples = len(X) + test_size
            for frac in split:
                it_count = int(frac * samples)
                split_data.append((X[start_idx:start_idx + it_count], Y[start_idx:start_idx + it_count]))
                start_idx += it_count
            return split_data

        else:
            raise ValueError("[GenericDatasource] Spliting values have to equal 1.0")

    def run_dir(self, path):
        """
        Multiprocess execution over directories of path in search for images.
        """

        if not os.path.isdir(path):
            return None

        dlist = []
        files = os.listdir(path)
        X, Y = ([], [])

        if self.multi_dir:
            for f in files:
                item = os.path.join(path, f)
                if os.path.isdir(item):
                    dlist.append(item)

            mdata = multiprocess_run(self._run_multiprocess, tuple(), dlist,
                                     self._cpu_count,
                                     step_size=1, output_dim=2, txt_label='directories', verbose=self._verbose)

        else:
            mdata = self._load_metadata_from_dir(self.path)

        X.extend(mdata[0])  # samples
        Y.extend(mdata[1])  # labels

        X, Y = self._shuffle(X, Y)
        return X, Y

    def _shuffle(self, X, Y):
        # Shuffle samples and labels maintaining relative order
        combined = list(zip(X, Y))
        # random.shuffle(combined)
        X[:], Y[:] = zip(*combined)

        return X, Y

    def split_metadata(self, split, data=None, test_size=0):
        """
        Returns all metadata split into N sets, defined by the spliting tuples

        @param data <tuple>
        @param split
        """
        # @param data <tuple>: (X,Y) if provided, split this sequence. Else, split full metadata
        if data is None:
            return self._split_data(split, self.X, self.Y, test_size)
        elif len(data) >= 2:
            return self._split_data(split, data[0], data[1], test_size)
        else:
            return None

    def load_metadata(self, metadata_file='metadata.pik'):
        """
        Iterates over data patches and creates an instance of a GenericImage subclass for each one
        Returns a tuples of lists (X,Y): X instances of GenericImage subclasses, Y labels;

        OBS: Dataset metadata is shuffled once here. Random sample generation is done during training.
        """

        X, Y = (None, None)
        reload_data = False
        reshuffle = False

        if self._cache.check_file_existence('split_ratio.pik'):
            split = self._cache.load('split_ratio.pik')
            if self._config.split != split:
                # Dump old data
                reshuffle = True
                if self.X is not None or self.Y is not None:
                    del self.X
                    del self.Y
                    self.X = None
                    self.Y = None

                if self._config.info:
                    print("[GenericDatasource] Previous split ratio {} is different from requested one {}. "
                          "Metadata will be reshuffled.".format(split, self._config.split))

        if self._cache.check_file_existence(metadata_file) and not reload_data:
            try:
                X, Y, name = self._cache.load(metadata_file)
            except ValueError:
                name = ''
                reload_data = True
            if name != self.name:
                reload_data = True

            if not reload_data and not reshuffle and self._verbose > 0:
                print("[GenericDatasource] Loaded data cache. Previously defined splitting can be used.")
        else:
            reload_data = True

        if reshuffle:
            X, Y = self._shuffle(X, Y)

        if reload_data:
            X, Y = self.run_dir(self.path)

        if reload_data or reshuffle:
            self._cache.dump((X, Y, self.name), metadata_file)
            self._cache.dump(tuple(self._config.split), 'split_ratio.pik')

        self.X = X.copy()
        self.Y = Y.copy()
        return np.asarray(X), np.asarray(Y)

    def load_data(self, split=None, keep_img=False, data=None):
        """
        Actually reads images and returns data ready for training
        Returns two tuples of NP arrays (X,Y): X data points, Y labels;

        @param split <tuple>
        @param keep_img <bool>
        @param data <tuple>
        """

        # @param split <tuple>: items are spliting fractions
        #
        #         If a spliting ratio is provided, return a list of tuples of size at most 3:
        #         1 - Train;
        #         2 - Validation;
        #         3 - Test;
        # @param keep_img <bool>: Keep image data in memory
        # @param data <tuple>: metadata defining images to load. If not provided, full dataset is used.

        if data is None and (self.X is None or self.Y is None):
            if self._verbose > 0:
                print("[GenericDatasource] Metadata not ready, loading...")
            self.load_metadata()

        # Which data to use?
        X, Y = None, None
        if data is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = data

        if self._config.pred_size > 0:
            samples = self._config.pred_size
        else:
            samples = len(X)
        y = np.array(Y[:samples], dtype=np.int32)
        if self._config.tdim is not None and len(self._config.tdim) == 2:
            img_dim = tuple(self._config.tdim) + (3,)
        else:
            dataset_dim = self.get_dataset_dimensions(X)[0]
            img_dim = dataset_dim[1:]
        X_data = np.zeros(shape=(samples,) + img_dim, dtype=np.float32)

        counter = 0
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=7)
        for i in range(samples):
            futures.append(executor.submit(X[i].read_image, keep_img, img_dim, self._verbose))

        if self._config.info:
            print("Reading images...")

        # for future in concurrent.futures.as_completed(futures):
        for i in range(samples):
            X_data[i] = futures[i].result()
            if self._verbose > 0:
                print(".", end='')

        if self._verbose > 0:
            print('\n')

        if split is None:
            return X_data, y
        else:
            return self._split_data(split, X_data, y)

    def sample_metadata(self, k, data=None, pos_rt=None, use_cache=True, s_idx=None):
        """
        Produces a sample of the full metadata with k items. Returns a cached sample if one exists

        @param k <int>
        @param k <float>
        @param data <tuple>
        @param pos_rt <float>
        @param use_cache <boolean>
        @param s_idx <np array>
        Return:
        - tuple (X,Y,sample_idx): X an Y have k elements, sample_idx has the indexes from X/Y that samples came from
        """

        # @param k <int>: total of samples
        # @param k <float>: percentile of the whole dataset
        # @param data <tuple>: data to sample from. If not given, use cached metadata
        # @param pos_rt <float>: load a percentile of positive samples
        # @param use_cache <boolean>: load cached data sample
        # @param s_idx <np array>: if given, return sample indexes with respect to data

        def regen_idx(s_x, X):
            samples = np.zeros(len(s_x), dtype=np.int32)
            print("Regenerating indexes...")
            xd = {}
            for i in range(len(X)):
                xd[X[i]] = i

            for k in range(len(s_x)):
                idx = xd.get(s_x[k], -1)
                if idx > 0:
                    samples[k] = idx
                else:
                    print("[GenericDataSource] Element not found: {}".format(s_x[k]))
            print("[GenericDatasource] Sample indexes regeneration: \n{}".format(samples))

            return samples

        # Init
        reload_data = False
        s_x, s_y = (None, None)
        samples = None

        if data is None and (self.X is None or self.Y is None):
            if self._config.verbose > 1:
                print("[GenericDatasource] Run load_metadata first!")
            return None
        elif data is not None:
            X, Y = data
        else:
            X, Y = self.X, self.Y

        # Check if we have the desired number of items
        if k <= 1.0:
            k = int(k * len(X))
        else:
            k = int(k)

        if self._cache.check_file_existence('sampled_metadata.pik') and use_cache:
            try:
                data = self._cache.load('sampled_metadata.pik')
            except ValueError:
                name = ''
                reload_data = True

            if len(data) == 3:
                s_x, s_y, name = data
            elif len(data) == 4:
                s_x, s_y, name, samples = data

            if name != self.name:
                reload_data = True

            if k != len(s_x):
                if self._config.info:
                    print("[GenericDatasource] Saved samples are different from requested ({} x {}). "
                          "Resampling...".format(k, len(s_x)))
                reload_data = True

            if not reload_data:
                if samples is None and self._config.spool > 0:
                    samples = regen_idx(s_x, X)
                if self._verbose > 0:
                    print("[GenericDatasource] Loaded sampled data cache. Previously defined splitting can be used.")

        else:
            reload_data = True

        if reload_data:

            # All operations are over indexes
            if pos_rt is not None:
                np_y = np.array(Y)
                unique, count = np.unique(np_y, return_counts=True)
                l_count = dict(zip(unique, count))
                pcount = min(int(pos_rt * k), l_count[1])
                pos_samples = np.random.choice(np.where(np_y == 1)[0], pcount, replace=False)
                neg_samples = np.random.choice(np.where(np_y != 1)[0], k - pcount, replace=False)
                samples = np.concatenate((pos_samples, neg_samples))
                np.random.shuffle(samples)
                del np_y
            else:
                samples = np.random.choice(range(len(X)), k, replace=False)

            s_x = [X[s] for s in samples]
            s_y = [Y[s] for s in samples]

        # Save last generated sample
        self._cache.dump((s_x, s_y, self.name, samples), 'sampled_metadata.pik')
        return np.asarray(s_x), np.asarray(s_y), samples


class PImage:
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
        self._path = path
        self._verbose = verbose
        self._keep = keep_img
        self._data = None
        self._dim = None
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

    def __eq__(self, other):
        if not isinstance(other, PImage):
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

        return data[y:(y + dy), x:(x + dx)]

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
