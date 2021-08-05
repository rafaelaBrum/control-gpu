#!/usr/bin/env python3
# -*- coding: utf-8

import os
import numpy as np
import random

from .CacheManager import CacheManager


def _split_origins(config, x_data, t_idx):
    """
    Separates patches of a predefined number of WSIs to be used as test set
    """

    cache_m = CacheManager()
    if cache_m.check_file_existence('testset.pik'):
        full_id, samples = cache_m.load('testset.pik')
        if samples is not None and config.info:
            print("[DataSetup] Using cached TEST SET. This is DANGEROUS. Use the metadata correspondent to the set.")
        return full_id, samples
            
    wsis = set()

    for k in x_data:
        wsis.add(k.getOrigin())

    # Defines slides to provide test set patches
    wsis = list(wsis)
    if config.wsilist is None:
        selected = set(random.choices(wsis, k=config.wsi_split))
    else:
        selected = set(config.wsilist)
    selected_idx = []

    if config.info:
        print("[DataSetup] WSIs selected to provide test patches:\n{}".format("\n".join(selected)))

    patch_count = {}
        
    for i in range(len(x_data)):
        w = x_data[i].getOrigin()
        if w in selected:
            patch_count.setdefault(w, [])
            patch_count[w].append(i)

    if config.wsimax is None or len(config.wsimax) != len(config.wsilist):
        for w in patch_count:
            if config.info:
                print("[Datasetup] Using all {} patches from slide {}".format(len(patch_count[w]), w))
            selected_idx.extend(patch_count[w])
    else:
        for i in range(len(config.wsilist)):
            w = config.wsilist[i]
            pc = int(config.wsimax[i] * len(patch_count[w]))
            pc = min(pc, len(patch_count[w]))
            selected_idx.extend(patch_count[w][:pc])
            if config.info:
                print("[Datasetup] Using {} ({:.2f}%) patches from slide {}".format(pc, 100*pc/len(patch_count[w]), w))

    t_idx = min(len(selected_idx), t_idx)
    samples = np.random.choice(selected_idx, t_idx, replace=False)
    full_id = np.asarray(selected_idx, dtype=np.int32)
    cache_m.dump((full_id, samples), 'testset.pik')
        
    return full_id, samples


def split_test(config, ds):

    # Test set is extracted from the last items of the full DS or from a test dir and is not changed for the whole run
    fX, fY = ds.load_metadata()
    test_x = None
    test_y = None
    
    tsp = config.split[-1:][0]
    t_idx = 0
    if tsp > 1.0:
        t_idx = int(tsp)
    else:
        t_idx = int(tsp * len(fX))

    # Configuration option that limits test set size
    t_idx = min(config.pred_size, t_idx) if config.pred_size > 0 else t_idx

    if config.testdir is None or not os.path.isdir(config.testdir):
        if config.wsi_split > 0 or config.wsilist is not None:
            full_id, samples = _split_origins(config, fX, t_idx)
            test_x = fX[samples]
            test_y = fY[samples]
            X = np.delete(fX, full_id)
            Y = np.delete(fY, full_id)
        else:
            test_x = fX[- t_idx:]
            test_y = fY[- t_idx:]
            X, Y = fX[:-t_idx], fY[:-t_idx]
        ds.check_paths(test_x, config.predst)
    else:
        x_test, y_test = ds.run_dir(config.testdir)
        t_idx = min(len(x_test), t_idx)
        samples = np.random.choice(len(x_test), t_idx, replace=False)
        test_x = [x_test[s] for s in samples]
        test_y = [y_test[s] for s in samples]
        del x_test
        del y_test
        del samples
        X, Y = fX, fY

    return test_x, test_y, X, Y
