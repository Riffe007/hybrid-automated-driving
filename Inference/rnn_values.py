import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class MDNRNN(models.Model):
    def __init__(self, hps):
        super(MDNRNN, self).__init__()
        self.hps = hps
        self.build_model()

    def build_model(self):
        # Configuring LSTM layer
        self.lstm = layers.LSTM(self.hps.rnn_size, return_sequences=True)

        # Layer normalization
        if self.hps.use_layer_norm:
            self.lstm = layers.LayerNormalization()(self.lstm)

        # Dropout layers
        self.input_dropout = layers.Dropout(self.hps.input_dropout_prob)
        self.output_dropout = layers.Dropout(self.hps.output_dropout_prob)

        # MDN layer
        self.mdn_dense = layers.Dense(self.hps.output_seq_width * self.hps.num_mixture * 3)

    def call(self, inputs, training=False):
        x = self.input_dropout(inputs, training=training)
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

# Define hyperparameters
hps = {
    'num_steps': 2000,
    'max_seq_len': 1000,
    'input_seq_width': 35,
    'output_seq_width': 32,
    'rnn_size': 256,
    'batch_size': 100,
    'grad_clip': 1.0,
    'num_mixture': 5,
    'learning_rate': 0.001,
    'decay_rate': 1.0,
    'min_learning_rate': 0.00001,
    'use_layer_norm': True,
    'use_recurrent_dropout': True,
    'recurrent_dropout_prob': 0.90,
    'use_input_dropout': True,
    'input_dropout_prob': 0.90,
    'use_output_dropout': True,
    'output_dropout_prob': 0.90,
    'is_training': True
}

# Instantiate and use the model
mdnrnn_model = MDNRNN(hps)
