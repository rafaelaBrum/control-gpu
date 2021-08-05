#!/usr/bin/env python3
# -*- coding: utf-8

# Training callbacks
from keras.callbacks import Callback

import numpy as np


class CalculateF1Score(Callback):
    """
    Calculates F1 score as a callback function. The right way to do it.
    NOTE: ONLY WORKS FOR BINARY CLASSIFICATION PROBLEM
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """

    def __init__(self, val_data, period=20, batch_size=32, info=True):
        """
        Use the same data generator that was provided as validation

        @param val_data <generator>: Should be some subclass of GenericIterator
        @param period <int>: calculate F1 and AUC each period epochs
        @param batch_size <int>: Batch
        @param info <boolean>: print progress information
        """
        super().__init__()
        self.val_data = val_data
        self.bsize = batch_size
        self.period = period
        self.info = info
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate F1 each X epochs
        """
        if (epoch+1) % self.period != 0:
            return None

        if not hasattr(self.val_data, 'classes') or self.val_data.classes > 2:
            return None
        
        from sklearn import metrics

        data_size = self.val_data.returnDataSize()
        Y_pred = np.zeros((data_size, self.val_data.classes), dtype=np.float32)
        Y = np.zeros((data_size, self.val_data.classes), dtype=np.int8)
        if data_size % self.bsize:
            stp = round((data_size / self.bsize) + 0.5)
        else:
            stp = data_size // self.bsize
        if self.info:
            print('[F1CB] Making batch predictions: ', end='')

        for i in range(stp):
            start_idx = i*self.bsize
            example = self.val_data.next()
            Y[start_idx:start_idx+self.bsize] = example[1]
            Y_pred[start_idx:start_idx+self.bsize] = self.model.predict_on_batch(example[0])
            if self.info:
                print(".", end='')

        print('')

        y_pred = np.argmax(Y_pred, axis=1)
        expected = np.argmax(Y, axis=1)

        f1 = metrics.f1_score(expected, y_pred, pos_label=1)
        print("F1 score: {0:.2f}".format(f1), end=' ')
        scores = Y_pred.transpose()[1]
    
        fpr, tpr, thresholds = metrics.roc_curve(expected, scores, pos_label=1)
        print("AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))

        del Y_pred
        del Y
        del y_pred
        del expected
