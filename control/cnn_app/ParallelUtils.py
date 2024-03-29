#!/usr/bin/env python3
# -*- coding: utf-8

import sys
import numpy as np
import multiprocessing

from tqdm import tqdm


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
            print("[{2}] Done transformations (step {0}/{1})".format(i, len(semaphores)-1, txt_label))
            sys.stdout.flush()

    # Free all possible memory
    pool.close()
    pool.join()

    del datapoints
    
    # remove None points
    return tuple(filter(lambda x: not x is None, datapoints_db))
