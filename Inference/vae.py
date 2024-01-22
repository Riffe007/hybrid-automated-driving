import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks

class ConvVAE:
    def __init__(self, input_dim=(64, 64, 3), z_dim=32, dense_size=1024):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.dense_size = dense_size
        self.encoder, self.decoder = self.build_models()

    def build_models(self):
        # Encoder
        encoder_inputs = layers.Input(shape=self.input_dim)
        x = self.build_conv_layers(encoder_inputs, is_encoder=True)
        z_mean = layers.Dense(self.z_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.z_dim, name="z_log_var")(x)
        z = layers.Lambda(self.sampling, output_shape=(self.z_dim,), name='z')([z_mean, z_log_var])
        encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = layers.Input(shape=(self.z_dim,), name='z_sampling')
        x = layers.Dense(self.dense_size, activation='relu')(latent_inputs)
        x = layers.Reshape((1, 1, self.dense_size))(x)
        decoder_outputs = self.build_conv_layers(x, is_encoder=False)
        decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

        return encoder, decoder

    def build_conv_layers(self, inputs, is_encoder):
        x = inputs
        conv_params = [
            {'filters': 32, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'},
            {'filters': 64, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'},
            {'filters': 64, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'},
            {'filters': 128, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'}
        ]

        if is_encoder:
            for layer_params in conv_params:
                x = layers.Conv2D(**layer_params)(x)
            x = layers.Flatten()(x)
        else:
            for layer_params in reversed(conv_params):
                x = layers.Conv2DTranspose(filters=layer_params['filters'], kernel_size=layer_params['kernel_size'], 
                                           strides=layer_params['strides'], activation=layer_params['activation'])(x)
        return x

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.z_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compile_vae(self):
        vae_inputs = self.encoder.input
        vae_outputs = self.decoder(self.encoder(vae_inputs)[2])
        vae = models.Model(vae_inputs, vae_outputs, name='vae')
        
        reconstruction_loss = losses.mean_squared_error(vae_inputs, vae_outputs)
        reconstruction_loss *= self.input_dim[0] * self.input_dim[1]
        kl_loss = 1 + self.encoder.get_layer("z_log_var").output - tf.square(self.encoder.get_layer("z_mean").output) - tf.exp(self.encoder.get_layer("z_log_var").output)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        return vae

    def train(self, data, epochs=10, batch_size=32):
        vae = self.compile_vae()
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        vae.fit(data, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop])

    def save_weights(self, filepath):
        self.vae.save_weights(filepath)

    def load_weights(self, filepath):
        self.vae.load_weights(filepath)

# Example usage
if __name__ == "__main__":
    vae = ConvVAE()
    # Load your data here
    # data = ...
    # vae.train(data)
    # vae.save_weights('path_to_save_weights')
