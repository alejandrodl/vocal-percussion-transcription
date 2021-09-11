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
                tf.keras.layers.Dense(12),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out


class VAE_Interim(tf.keras.Model):

    def __init__(self, latent_dim):
        super(VAE_Interim, self).__init__()
        self.latent_dim = 32
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
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


class CNN_Interim(tf.keras.Model):

    def __init__(self, num_labels, latent_dim):
        super(CNN_Interim, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.cnn(x)
        return out


def CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr) -> tf.keras.Model:
    """
    This method builds and returns a Model
    :return:
    """

    x_input = tf.keras.Input(shape=(64, 64, 1), dtype='float32')

    x = tf.keras.layers.InputLayer(input_shape=(64, 64, 1))(x_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Activation(activation='relu')(x)
    
    x_onset = tf.keras.layers.Dense(num_onset, activation='softmax', name="onset")(x)
    x_nucleus = tf.keras.layers.Dense(num_nucleus, activation='softmax', name="nucleus")(x)
    
    optimizer = tf.keras.optimizers.Adam(lr)

    model = tf.keras.Model(inputs=[x_input], outputs=[x_onset, x_nucleus])
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss={"onset":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        "nucleus":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)})
    
    return model



class CNN_Interim_Siamese(tf.keras.Model):

    def __init__(self, num_labels, latent_dim):
        super(CNN_Interim_Siamese, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim)
            ]
        )

    def call(self, x):

        cutoff = tf.shape(x)[2]//3

        x_anchor = x[:,:,:cutoff,:]
        x_positive = x[:,:,cutoff:2*cutoff,:]
        x_negative = x[:,:,2*cutoff:,:]

        out_anchor = self.cnn(x_anchor)
        out_positive = self.cnn(x_positive)
        out_negative = self.cnn(x_negative)

        return out_anchor, out_positive, out_negative