import tensorflow as tf
import tensorflow_probability as tfp



class VAE_Interim(tf.keras.Model):

    def __init__(self, latent_dim):
        super(VAE_Interim, self).__init__()
        self.latent_dim = 32
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim+self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=2*2*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(2, 2, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(3,3), strides=(2,2), padding='same'),
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

    def reparameterize(self, mean, logvar, batch_size):
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False): # apply_sigmoid=False
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, batch_size)
        out = self.decode(z)
        return out


model = VAE_Interim(32)
model.built = True
print(model.encoder.summary())
print(model.decoder.summary())