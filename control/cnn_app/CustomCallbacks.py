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
        print("F1 score: {0:f}".format(f1), end=' ')
        scores = Y_pred.transpose()[1]
    
        fpr, tpr, thresholds = metrics.roc_curve(expected, scores, pos_label=1)
        print("AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))

        del Y_pred
        del Y
        del y_pred
        del expected


class CalculateTestMetrics(Callback):
    """
    Calculates metrics on test dataset as a callback function. The right way to do it.
    NOTE: ONLY WORKS FOR BINARY CLASSIFICATION PROBLEM
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """

    def __init__(self, test_data, period=30, batch_size=64, info=True):
        """
        Use the data generator that was provided as test

        @param test_data <generator>: Should be some subclass of GenericIterator
        @param period <int>: calculate metrics each period epochs
        @param batch_size <int>: Batch
        @param info <boolean>: print progress information
        """
        super().__init__()
        self.test_data = test_data
        self.bsize = batch_size
        self.period = period
        self.info = info

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate F1 each X epochs
        """
        if (epoch + 1) % self.period != 0:
            return None

        if not hasattr(self.test_data, 'classes') or self.test_data.classes > 2:
            return None

        classes = self.test_data.classes

        from sklearn import metrics

        data_size = self.test_data.returnDataSize()
        Y_pred = np.zeros((data_size, self.test_data.classes), dtype=np.float32)
        Y = np.zeros((data_size, self.test_data.classes), dtype=np.int8)
        if data_size % self.bsize:
            stp = round((data_size / self.bsize) + 0.5)
        else:
            stp = data_size // self.bsize
        if self.info:
            print('[F1CB] Making batch predictions: ', end='')

        for i in range(stp):
            start_idx = i * self.bsize
            example = self.test_data.next()
            Y[start_idx:start_idx + self.bsize] = example[1]
            Y_pred[start_idx:start_idx + self.bsize] = self.model.predict_on_batch(example[0])
            if self.info:
                print(".", end='')

        print('')

        y_pred = np.argmax(Y_pred, axis=1)
        expected = np.argmax(Y, axis=1)

        stp = len(expected)
        # x is expected, y is actual
        m_conf = np.zeros((classes + 3, classes + 1))
        for i in range(stp):
            m_conf[expected[i]][y_pred[i]] = m_conf[expected[i]][y_pred[i]] + 1
        m_conf_2 = m_conf.tolist()
        # Total predictions and expectations for each class
        for i in range(classes):
            m_conf_2[classes][i] = "{0:.0f}".format(
                sum(m_conf.transpose()[i]))
            m_conf_2[i][classes] = "{0:.0f}".format(sum(m_conf[i]))
        # Correct rate
        for i in range(classes):
            m_conf_2[classes + 1][i] = "{0:.0f}/{1:.0f}".format(
                m_conf[i][i], sum(m_conf.transpose()[i]))
        # Accuracy
        for i in range(classes):
            m_conf_2[classes + 2][i] = "{0:.4f}".format(
                m_conf[i][i] / sum(m_conf.transpose()[i]))

        # Copying m_conf2 to m_conf
        for i in range(classes):
            m_conf[classes][i] = m_conf_2[classes][i]
            m_conf[i][classes] = m_conf_2[i][classes]
            m_conf[classes + 2][i] = m_conf_2[classes + 2][i]

        # Total samples
        m_conf_2[classes][classes] = "{0:.0f}".format(m_conf.sum())
        m_conf_2[classes + 1][classes] = ''
        m_conf_2[classes + 2][classes] = '{0:.2f}'.format(sum(np.diag(m_conf)) / m_conf.sum())
        # Store accuracy in m_conf also
        m_conf[classes + 2][classes] = m_conf_2[classes + 2][classes]

        scores = Y_pred.transpose()[1]
        fpr, tpr, thresholds = metrics.roc_curve(expected, scores, pos_label=1)

        print("AUC: {0:f}".format(metrics.roc_auc_score(expected, scores)))

        print("Accuracy: {0:.3f}".format(m_conf[classes + 2][classes]))

        print("m_conf", m_conf)

        neg_accuracy = m_conf[4][0]
        pos_accuracy = m_conf[4][1]
        precision = m_conf[1][1] / m_conf[2][1]
        recall = m_conf[1][1] / m_conf[1][2]
        f1_score = 2 * m_conf[1][1] / (m_conf[1][2] + m_conf[2][1])

        print("False positive rates: {0}".format(fpr))
        print("True positive rates: {0}".format(tpr))
        print("Thresholds: {0}".format(thresholds))
        print(f"Accuracy: {m_conf[nclasses + 2][nclasses]}")
        print(f"Negative Accuracy: {neg_accuracy}")
        print(f"Positive Accuracy: {pos_accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        del Y_pred
        del Y
        del y_pred
        del expected