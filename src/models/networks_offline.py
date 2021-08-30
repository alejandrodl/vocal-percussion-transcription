import pdb
import tensorflow as tf
import tensorflow_probability as tfp



class MLP_Processed(tf.keras.Model):

    def __init__(self, num_labels, input_length):
        super(MLP_Processed, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(8),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out



class MLP_Interim(tf.keras.Model):

    def __init__(self, num_labels, input_length):
        super(MLP_Interim, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(16),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out
      
    
class CVAE_External(tf.keras.Model):


    def __init__(self, latent_dim, batch_size, kernel_height, kernel_width, dropout):
        super(CVAE_External, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32, 64, 1)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=(kernel_height,kernel_width), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=(kernel_height,kernel_width), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(kernel_height,kernel_width), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(kernel_height,kernel_width), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(kernel_height,kernel_width), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(kernel_height,kernel_width), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.relu),
                # No activation
                tf.keras.layers.Dense(latent_dim+latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1*2*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(1, 2, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(kernel_height,kernel_width), strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(kernel_height,kernel_width), strides=1, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=(kernel_height,kernel_width), strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=(kernel_height,kernel_width), strides=1, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(kernel_height,kernel_width), strides=2, padding='same'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(kernel_height,kernel_width), strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False): # apply_sigmoid=False
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out