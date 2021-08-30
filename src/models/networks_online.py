import pdb
import tensorflow as tf
import tensorflow_probability as tfp


class MLP_RT(tf.keras.Model):

    def __init__(self, num_labels, input_length, dropout):
        super(MLP_RT, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out
    
    
class MLP_RT_Big(tf.keras.Model):

    def __init__(self, num_labels, input_length, dropout):
        super(MLP_RT_Big, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out
    
    
    
class CNN_1D(tf.keras.Model):

    def __init__(self, kernel_length, num_labels, input_length, dropout):
        super(CNN_1D, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                #tf.keras.layers.Conv1D(
                    #filters=4, kernel_size=kernel_length, padding='same', activation='relu'),
                #tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=8, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=16, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=64, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out
    
    
    
class CNN_1D_Big(tf.keras.Model):

    def __init__(self, kernel_length, num_labels, input_length, dropout):
        super(CNN_1D_Big, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                #tf.keras.layers.Conv1D(
                    #filters=4, kernel_size=kernel_length, padding='same', activation='relu'),
                #tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=8, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=8, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=16, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=16, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Conv1D(
                    filters=64, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=64, kernel_size=kernel_length, padding='same', activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=3),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out