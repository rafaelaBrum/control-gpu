#!/usr/bin/env python3
# -*- coding: utf-8

import importlib
import os
from tqdm import tqdm
import numpy as np
import queue
import concurrent.futures
from threading import Thread

from .CellRep import CellRep
from .BatchGenerator import ThreadedGenerator
from .DataSetup import split_test
from . import Exitcodes
from .CacheManager import CacheManager
from .Output import PrintConfusionMatrix
# from AL.Common import load_model_weights

# Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.utils import to_categorical
from keras.models import load_model

# Scikit learn
from sklearn import metrics


def _fill_queue(queue, steps, generator, workers=3):
    qw = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    futures = {}
    for i in range(steps):
        futures[qw.submit(generator.next)] = i

    for f in concurrent.futures.as_completed(futures):
        queue.put((f.result(), futures[f]), block=True)
        

def run_prediction(config, locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting prediction process....")

    if locations is not None:
        cache_m = CacheManager(locations=locations)
    if config.print_pred:
        print_prediction(config)
    else:
        # build_ensemble = True if config.strategy == "EnsembleTrainer" else False
        build_ensemble = False
        predictor = Predictor(config, build_ensemble=build_ensemble)
        predictor.run()


def print_prediction(config, target=True):
    cache_m = CacheManager()

    if not os.path.isfile(cache_m.file_location('test_pred.pik')):
        return None
    
    # Load predictions
    (expected, Y_pred, nclasses) = cache_m.load('test_pred.pik')
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Output metrics
    if nclasses > 2:
        f1 = metrics.f1_score(expected, y_pred, average='weighted')
    else:
        f1 = metrics.f1_score(expected, y_pred, pos_label=1)
    print("F1 score: {0:.2f}".format(f1))

    m_conf = PrintConfusionMatrix(y_pred, expected, nclasses, config, "TILs")

    # ROC AUC
    # Get positive scores (binary only)
    if nclasses == 2:
        scores = Y_pred.transpose()[1]
        fpr, tpr, thresholds = metrics.roc_curve(expected, scores, pos_label=1)
        if target:
            print("AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))
        else:
            print("FN AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))

    if target:
        print("Accuracy: {0:.3f}".format(m_conf[nclasses+2][nclasses]))
    else:
        print("FN Accuracy: {0:.3f}".format(m_conf[nclasses+2][nclasses]))
        
    if config.verbose > 1:
        print("False positive rates: {0}".format(fpr))
        print("True positive rates: {0}".format(tpr))
        print("Thresholds: {0}".format(thresholds))


class Predictor(object):
    """
    Class responsible for running the predictions and outputing results
    """

    def __init__(self, config, keepImg=False, **kwargs):
        """
        @param config <parsed configurations>: configurations

        Optional keyword arguments:
        @param build_ensemble <boolean>: ask for an ensemble model
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None
        self._keep = keepImg

        if 'build_ensemble' in kwargs:
            self._ensemble = kwargs['build_ensemble']
        else:
            self._ensemble = False

    def run(self, x_test=None, y_test=None, load_full=True, net_model=None, target=True):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.

        If provided x_test and y_test data, runs prediction with them.

        @param load_full <boolean>: loads full model with load_model function. If ensemble, load individual
        model weights
        @param net_model <GenericModel subclass>: performs predictions with this model
        @param target <boolean>: Target network predicitons?
        """
        net_name = self._config.network
        if net_name is None or net_name == '':
            print("A network should be specified")
            return Exitcodes.RUNTIME_ERROR

        # Load DS when a prediction only run is being made
        if net_model is not None:
            self._ds = net_model.get_ds()
        elif self._config.data:
            dsm = importlib.import_module('control.cnn_app', self._config.data)
            if self._config.testdir:
                self._ds = getattr(dsm, self._config.data)(self._config.testdir, self._config.keepimg, self._config)
            else:
                self._ds = getattr(dsm, self._config.data)(self._config.predst, self._config.keepimg, self._config)
        else:
            self._ds = CellRep(self._config.predst, self._config.keepimg, self._config)
            
        if net_model is None:
            net_module = importlib.import_module('control.cnn_app', net_name)
            net_model = getattr(net_module, net_name)(self._config, self._ds)

        if x_test is None or y_test is None:
            x_test, y_test, _, _ = split_test(self._config, self._ds)

        self.run_test(net_model, x_test, y_test, load_full, target)
        
    def run_test(self, model, x_test, y_test, load_full=True, target=True):
        """
        This should be executed after a model has been trained
        """

        cache_m = CacheManager()
        split = None
        if os.path.isfile(cache_m.file_location('split_ratio.pik')):
            split = cache_m.load('split_ratio.pik')
        else:
            print("[Predictor] A previously trained model and dataset should exist. No previously defined spliting "
                  "found.")
            return Exitcodes.RUNTIME_ERROR

        if self._config.verbose > 0:
            unique, count = np.unique(y_test, return_counts=True)
            l_count = dict(zip(unique, count))
            if target:
                print("\n[Predictions] Starting target net evaluation")
            else:
                print("\n[Predictions] Starting prediction phase")
            if len(unique) > 2:
                print("Test items:")
                print("\n".join(["label {0}: {1} items" .format(key, l_count[key]) for key in unique]))
            else:
                if  1 not in l_count:
                    l_count[1] = 0
                print("Test labels: {0} are 0; {1} are 1;\n "
                      "- {2:.2f} are positives".format(l_count.get(0, 0), l_count.get(1, 0),
                                                       (l_count.get(1, 0)/(l_count.get(0, 1)+l_count.get(1, 1)))))
            print("Test set: {} items".format(len(y_test)))

        if self._ensemble or self._config.delay_load:
            X, Y = x_test, y_test
        else:
            X, Y = self._ds.load_data(data=(x_test, y_test), keepImg=self._keep)
                        
        if self._config.verbose > 1:
            print("Y original ({1}):\n{0}".format(Y, Y.shape))        

        # session setup
        sess = K.get_session()
        
        if self._ensemble:
            # Weights should be loaded during ensemble build
            if hasattr(model, 'build_ensemble'):
                single, parallel = model.build_ensemble(training=False, npfile=True, new=False, load_weights=load_full)
                if parallel:
                    if self._config.info:
                        print("Using multigpu model for predictions.")
                    pred_model = parallel
                else:
                    pred_model = single
            else:
                if self._config.info:
                    print('[Predictor] Model not prepared to build ensembles, implement or choose another model')
                return None
        elif load_full and os.path.isfile(model.get_model_cache()):
            try:
                pred_model = load_model(model.get_model_cache())
                if self._config.info:
                    print("Model loaded from: {0}".format(model.get_model_cache()))
            except ValueError:
                pred_model, _ = model.build(training=False, pre_load_w=False)
                pred_model.load_weights(model.get_weights_cache())
                if self._config.info:
                    print("Model weights loaded from: {0}".format(model.get_weights_cache()))

        bsize = 2*self._config.batch_size
        stp = int(np.ceil(len(X) / bsize))

        image_generator = ImageDataGenerator(samplewise_center=self._config.batch_norm, 
                                             samplewise_std_normalization=self._config.batch_norm)

        if self._ensemble or self._config.delay_load:
            fix_dim = model.check_input_shape()

            test_generator = ThreadedGenerator(dps=(X, Y),
                                               classes=self._ds.nclasses,
                                               dim=fix_dim,
                                               batch_size=bsize,
                                               image_generator=image_generator,
                                               extra_aug=self._config.augment,
                                               shuffle=False,
                                               verbose=self._verbose,
                                               input_n=self._config.emodels if self._ensemble else 1,
                                               keep=self._keep)
        else:
            Y = to_categorical(Y, self._ds.nclasses)
            test_generator = image_generator.flow(x=X,
                                                  y=Y,
                                                  batch_size=bsize,
                                                  shuffle=False)

        del Y
        
        if self._config.progressbar:
            l = tqdm(desc="Making predictions...", total=stp)

        # Multi-threaded batch queue setup
        Y_pred = np.zeros((len(X), self._ds.nclasses), dtype=np.float32)
        expected = np.zeros((len(X), self._ds.nclasses), dtype=np.int32)
        q = queue.Queue(maxsize=self._config.cpu_count*2)
        th = Thread(target=_fill_queue, name="Batch loader", args=(q, stp, test_generator, self._config.cpu_count))
        th.start()
        for k in range(stp):
            example, i = q.get(block=True)
            start_idx = i*bsize
            # example = test_generator.next()
            with sess.as_default():
                with sess.graph.as_default():
                    Y_pred[start_idx:start_idx+bsize] = pred_model.predict_on_batch(example[0])
                    expected[start_idx:start_idx+bsize] = example[1]
            if self._config.progressbar:
                l.update(1)
            elif self._config.info:
                print("Batch prediction ({0}/{1})".format(i, stp))

        del X
        del test_generator
        th.join()
        
        if self._config.progressbar:
            l.close()

        y_pred = np.argmax(Y_pred, axis=1)
        expected = np.argmax(expected, axis=1)

        if self._config.verbose > 0:
            if self._config.verbose > 1:
                np.set_printoptions(threshold=np.inf)
                print("Predicted probs ({1}):\n{0}".format(Y_pred, Y_pred.shape))
            # print("Y ({1}):\n{0}".format(Y, Y.shape))
            print("expected ({1}):\n{0}".format(expected, expected.shape))
            print("Predicted ({1}):\n{0}".format(y_pred, y_pred.shape))
            
        # Save predictions
        cache_m.dump((expected, Y_pred, self._ds.nclasses), 'test_pred.pik')

        # Output metrics
        print_prediction(self._config, target)
