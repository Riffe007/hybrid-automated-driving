import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class MDNRNN(models.Model):
    def __init__(self, hps, **kwargs):
        super(MDNRNN, self).__init__(**kwargs)
        self.hps = hps
        self.num_mixture = self.hps['num_mixture']
        self.output_seq_width = self.hps['output_seq_width']
        self.rnn_size = self.hps['rnn_size']
        self.input_dropout_prob = self.hps['input_dropout_prob']
        self.output_dropout_prob = self.hps['output_dropout_prob']
        self.use_layer_norm = self.hps['use_layer_norm']

        self.build_model()

    def build_model(self):
        # LSTM layer
        self.lstm = layers.LSTM(self.rnn_size, return_sequences=True)
        if self.use_layer_norm:
            self.lstm = layers.LayerNormalization()(self.lstm)

        # Dropout layers
        self.input_dropout = layers.Dropout(self.input_dropout_prob)
        self.output_dropout = layers.Dropout(self.output_dropout_prob)

        # Mixture Density Network (MDN) components
        self.mdn_dense = layers.Dense(self.output_seq_width * self.num_mixture * 3)

    def call(self, x, training=False):
        x = self.input_dropout(x, training=training)
        rnn_output = self.lstm(x)
        rnn_output = self.output_dropout(rnn_output, training=training)
        mdn_output = self.mdn_dense(rnn_output)
        return self.split_mdn_outputs(mdn_output)

    def split_mdn_outputs(self, mdn_output):
        logmix, mean, logstd = tf.split(mdn_output, 3, axis=-1)
        logmix = logmix - tf.reduce_logsumexp(logmix, axis=-1, keepdims=True)
        return logmix, mean, logstd

    def compute_loss(self, y_true, y_pred):
        logmix, mean, logstd = y_pred
        flat_target_data = tf.reshape(y_true, [-1, 1])
        loss = self.get_lossfunc(logmix, mean, logstd, flat_target_data)
        return tf.reduce_mean(loss)

    @staticmethod
    def get_lossfunc(logmix, mean, logstd, y):
        v = logmix + MDNRNN.tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, axis=1, keepdims=True)
        return -tf.reduce_mean(v)

    @staticmethod
    def tf_lognormal(y, mean, logstd):
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - np.log(np.sqrt(2.0 * np.pi))

# Hyperparameters
hps = {
    'num_mixture': 5,
    'output_seq_width': 32,
    'rnn_size': 256,
    'input_dropout_prob': 0.2,
    'output_dropout_prob': 0.2,
    'use_layer_norm': True,
    # Additional hyperparameters
}

# Instantiate and use the model
mdnrnn_model = MDNRNN(hps)

