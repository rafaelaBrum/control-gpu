import tensorflow as tf

from model_daemon import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense1 = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=2048, activation=tf.nn.relu)
        dense3 = tf.layers.dense(inputs=dense2, units=2048, activation=tf.nn.relu)
        dense4 = tf.layers.dense(inputs=dense3, units=2048, activation=tf.nn.relu)
        dense5 = tf.layers.dense(inputs=dense4, units=2048, activation=tf.nn.relu)
        dense6 = tf.layers.dense(inputs=dense5, units=2048, activation=tf.nn.relu)
        dense7 = tf.layers.dense(inputs=dense6, units=2048, activation=tf.nn.relu)
        dense8 = tf.layers.dense(inputs=dense7, units=2048, activation=tf.nn.relu)
        dense9 = tf.layers.dense(inputs=dense8, units=2048, activation=tf.nn.relu)
        dense10 = tf.layers.dense(inputs=dense9, units=2048, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense10, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # TODO: Confirm that opt initialized once is ok?
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
