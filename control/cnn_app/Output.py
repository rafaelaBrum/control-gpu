#!/usr/bin/env python3
# -*- coding: utf-8
import os
import numpy as np

from pandas import DataFrame


def PrintConfusionMatrix(y_pred, expected, classes, args, label):
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
        m_conf_2[classes + 2][i] = "{0:.2f}".format(
            m_conf[i][i] / sum(m_conf.transpose()[i]))

    # Total samples
    m_conf_2[classes][classes] = "{0:.0f}".format(m_conf.sum())
    m_conf_2[classes + 1][classes] = ''
    m_conf_2[classes + 2][classes] = '{0:.2f}'.format(sum(np.diag(m_conf)) / m_conf.sum())
    # Store accuracy in m_conf also
    m_conf[classes + 2][classes] = m_conf_2[classes + 2][classes]

    col = [i for i in range(classes)] + ['Expected Total']
    ind = [i for i in range(classes)] + \
          ['Predicted Total', 'Correct Rate', 'Accuracy']

    if args.info:
        df = DataFrame(m_conf_2, columns=col, index=ind)
        print("Confusion matrix ({0}):".format(label))
        print(df)
        print('\n')

        fd = open(
            os.path.join(os.path.abspath(args.logdir), 'confusion_matrix_{0}-nn{1}.csv'.format(label, args.network)),
            'w')
        df.to_csv(fd, columns=col, index=ind, header=True)
        fd.close()

    return m_conf
