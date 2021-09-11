import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append('/homes/adl30/vocal-percussion-transcription')
from src.utils import *
from networks_offline import *



class EarlyStopping_Phoneme(tf.keras.callbacks.Callback):
    '''
    Function for early stopping for phoneme labels. It considers both the onset and nucleus losses.
    '''
    def __init__(self, patience=0):
        super(EarlyStopping_Phoneme, self).__init__()

        self.patience = patience
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = 0

    def on_epoch_end(self, epoch, logs=None):

        onset_loss = logs.get('val_onset_accuracy')
        nucleus_loss = logs.get('val_nucleus_accuracy')

        if np.greater(0.6*onset_loss+0.4*nucleus_loss, self.best_loss):
            self.best_loss = 0.6*onset_loss+0.4*nucleus_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Global parameters

percentage_train = 85
#modes = ['vae','classall','classred','syllall','syllred','phonall','phonred']
modes = ['siamese']
min_acc = [0.53,0.66,0.3,0.35,0.55,0.6]

# Data parameters

frame_sizes = ['1024']

# Network parameters

latent_dim = 32

# Training parameters

epochs = 10000
patience_lr = 5
patience_early = 10
batch_size = 512

# Class weighting

onset_loss_weight = 0.6
nucleus_loss_weight = 0.4
class_weight = {'onset': onset_loss_weight, 'nucleus': nucleus_loss_weight}

# Normalisation values

norm_min_max_1 = np.load('../../data/offline_norm_min_max_1.npy')
norm_min_max_2 = np.load('../../data/offline_norm_min_max_2.npy')

# Main loop

for m in range(len(modes)):

    mode = modes[m]

    validation_losses_mode = np.zeros((len(frame_sizes),28))

    if not os.path.isdir('../../models/' + mode):
        os.mkdir('../../models/' + mode)

    if not os.path.isdir('../../data/processed/' + mode):
        os.mkdir('../../data/processed/' + mode)

    for a in range(len(frame_sizes)):

        frame_size = frame_sizes[a]

        for part in range(28):

            print('\n')
            print([mode,frame_size,part])
            print('\n')

            # Spectrogram loading

            pretrain_dataset = np.zeros((1, 64, 64))
            for n in range(28):
                if n==part:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '_' + frame_size + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_0' + str(n) + '_' + frame_size + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(n) + '_' + frame_size + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_' + str(n) + '_' + frame_size + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

            # Spectrogram normalisation

            pretrain_dataset = (pretrain_dataset-norm_min_max_1[a][0])/(norm_min_max_1[a][1]-norm_min_max_1[a][0]+1e-16)
            pretrain_dataset = np.log(pretrain_dataset+1e-4)
            pretrain_dataset = (pretrain_dataset-norm_min_max_2[a][0])/(norm_min_max_2[a][1]-norm_min_max_2[a][0]+1e-16)

            # Data preparation

            cutoff_train = int((percentage_train/100)*pretrain_dataset.shape[0])
            pretrain_dataset_train = pretrain_dataset[:cutoff_train]
            pretrain_dataset_test = pretrain_dataset[cutoff_train:]

            pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1).astype('float32')
            pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1).astype('float32')

            print(pretrain_dataset_train.shape)
            print(pretrain_dataset_test.shape)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_train)

            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_test)

            # Load and process classes

            if 'syllall' in mode or 'phonall' in mode:

                classes_onset = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
                classes_onset = classes_onset[1:]

                classes_nucleus = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
                classes_nucleus = classes_nucleus[1:]

                train_classes_onset = classes_onset[:cutoff_train].astype('float32')
                train_classes_nucleus = classes_nucleus[:cutoff_train].astype('float32')
                test_classes_onset = classes_onset[cutoff_train:].astype('float32')
                test_classes_nucleus = classes_nucleus[cutoff_train:].astype('float32')

                num_onset = np.max(classes_onset)+1
                num_nucleus = np.max(classes_nucleus)+1

            elif 'syllred' in mode or 'phonred' in mode:

                classes_onset = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
                classes_onset = classes_onset[1:]

                classes_nucleus = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
                classes_nucleus = classes_nucleus[1:]

                num_onset = np.max(classes_onset)+1
                num_nucleus = np.max(classes_nucleus)+1

            elif 'classall' in mode:

                classes_str = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
                classes_str = classes_str[1:]

                classes = np.zeros(len(classes_str))
                for n in range(len(classes_str)):
                    if classes_str[n]=='kd':
                        classes[n] = 0
                    elif classes_str[n]=='sd':
                        classes[n] = 1
                    elif classes_str[n]=='hhc':
                        classes[n] = 2
                    elif classes_str[n]=='hho':
                        classes[n] = 3

                num_classes = np.max(classes)+1

            elif 'classred' in mode:

                classes_str = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
                classes_str = classes_str[1:]

                classes = np.zeros(len(classes_str))
                for n in range(len(classes_str)):
                    if classes_str[n]=='kd':
                        classes[n] = 0
                    elif classes_str[n]=='sd':
                        classes[n] = 0
                    elif classes_str[n]=='hhc':
                        classes[n] = 1
                    elif classes_str[n]=='hho':
                        classes[n] = 1

                num_classes = np.max(classes)+1

            elif 'siamese' in mode:

                classes = np.zeros(1)

                for n in range(28):
                    if n==part:
                        continue
                    else:
                        classes_pre_siamese = np.zeros(1)
                        pretrain_dataset = np.zeros((1, 64, 64))
                        if n<=9:
                            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '_' + frame_size + '.npy')))
                            classes_str = np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')
                            classes_pre = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='kd':
                                    classes_pre_siamese[n] = (part*4)
                                elif classes_str[n]=='sd':
                                    classes_pre_siamese[n] = (part*4)+1
                                elif classes_str[n]=='hhc':
                                    classes_pre_siamese[n] = (part*4)+2
                                elif classes_str[n]=='hho':
                                    classes_pre_siamese[n] = (part*4)+3
                            classes_pre_siamese = np.vstack((classes_pre_siamese, classes_pre))
                        else:
                            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(n) + '_' + frame_size + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_pre = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='kd':
                                    classes_pre_siamese[n] = (part*4)
                                elif classes_str[n]=='sd':
                                    classes_pre_siamese[n] = (part*4)+1
                                elif classes_str[n]=='hhc':
                                    classes_pre_siamese[n] = (part*4)+2
                                elif classes_str[n]=='hho':
                                    classes_pre_siamese[n] = (part*4)+3
                            classes_pre_siamese = np.vstack((classes_pre_siamese, classes_pre))
                        classes_str = classes_str[1:]
                        pretrain_dataset = pretrain_dataset[1:]

                

                                


                num_classes = np.max(classes)+1

            if 'syll' in mode:

                combinations = []
                classes = np.zeros(len(classes_onset))
                for n in range(len(classes_onset)):
                    combination = [classes_onset[n],classes_nucleus[n]]
                    if combination not in combinations:
                        combinations.append(combination)
                        classes[n] = combinations.index(combination)
                    else:
                        classes[n] = combinations.index(combination)

                num_classes = np.max(classes)+1
                
                pretrain_classes_train = classes[:cutoff_train].astype('float32')
                pretrain_classes_test = classes[cutoff_train:].astype('float32')

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test)

            elif 'phon' in mode:
                
                pretrain_classes_train_onset = classes_onset[:cutoff_train].astype('float32')
                pretrain_classes_train_nucleus = classes_nucleus[:cutoff_train].astype('float32')
                pretrain_classes_test_onset = classes_onset[cutoff_train:].astype('float32')
                pretrain_classes_test_nucleus = classes_nucleus[cutoff_train:].astype('float32')

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train_onset)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train_nucleus)

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test_onset)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test_nucleus)

            elif 'class' in mode:

                pretrain_classes_train = classes[:cutoff_train].astype('float32')
                pretrain_classes_test = classes[cutoff_train:].astype('float32')

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test)

            # Train models

            if mode=='vae':

                while validation_losses_mode[a,part] > 0.003:

                    model = VAE_Interim(latent_dim)

                    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

                    with tf.device(gpu_name):

                        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=False)
                        history = model.fit(pretrain_dataset_train, pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_dataset_test), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)  # , verbose=0
                        validation_losses_mode[a,part] = history.history['val_loss'][-10]
                        print(validation_losses_mode[a,part])

            elif 'phon' in mode:

                while validation_losses_mode[a,part] < min_acc[m-1]::

                    model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr=3*1e-4)

                    early_stopping = EarlyStopping_Phoneme(patience=7)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_onset_accuracy', patience=14)

                    with tf.device(gpu_name):

                        history = model.fit(pretrain_dataset_train, [pretrain_classes_train_onset, pretrain_classes_train_nucleus], batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,[pretrain_classes_test_onset,pretrain_classes_test_nucleus]), callbacks=[early_stopping,lr_scheduler], class_weight=class_weight, shuffle=True, verbose=0)  # , verbose=0
                        validation_losses_mode[a,part] = (history.history['val_onset_accuracy'][-10]+history.history['val_nucleus_accuracy'][-10])/2
                        print(validation_losses_mode[a,part])

            else:

                while validation_losses_mode[a,part] < min_acc[m-1]:

                    model = CNN_Interim(num_classes, latent_dim)

                    optimizer = tf.keras.optimizers.Adam(lr=3*1e-4)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr)

                    with tf.device(gpu_name):

                        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                        history = model.fit(pretrain_dataset_train, pretrain_classes_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_classes_test), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)  # , verbose=0
                        validation_losses_mode[a,part] = history.history['val_accuracy'][-10]
                        print(validation_losses_mode[a,part])

            if part<=9:
                model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + frame_size + '_0' + str(part) + '.h5')
            else:
                model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + frame_size + '_' + str(part) + '.h5')

            # Compute processed features

            print('Computing features...')

            if part<=9:
                train_dataset = np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(part) + '_' + frame_size + '.npy')
                test_dataset = np.load('../../data/interim/AVP/Dataset_Test_0' + str(part) + '_' + frame_size + '.npy')
            else:
                train_dataset = np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(part) + '_' + frame_size + '.npy')
                test_dataset = np.load('../../data/interim/AVP/Dataset_Test_' + str(part) + '_' + frame_size + '.npy')

            train_dataset = (train_dataset-norm_min_max_1[a][0])/(norm_min_max_1[a][1]-norm_min_max_1[a][0]+1e-16)
            train_dataset = np.log(train_dataset+1e-4)
            train_dataset = (train_dataset-norm_min_max_2[a][0])/(norm_min_max_2[a][1]-norm_min_max_2[a][0]+1e-16)

            test_dataset = (test_dataset-norm_min_max_1[a][0])/(norm_min_max_1[a][1]-norm_min_max_1[a][0]+1e-16)
            test_dataset = np.log(test_dataset+1e-4)
            test_dataset = (test_dataset-norm_min_max_2[a][0])/(norm_min_max_2[a][1]-norm_min_max_2[a][0]+1e-16)

            train_dataset = np.expand_dims(train_dataset,axis=-1).astype('float32')
            test_dataset = np.expand_dims(test_dataset,axis=-1).astype('float32')

            if 'phon' in mode:
                extractor = tf.keras.Sequential()
                for layer in model.layers[:-4]:
                    extractor.add(layer)
                extractor.built = True
                train_features = extractor.predict(train_dataset)
                test_features = extractor.predict(test_dataset)
            elif mode=='vae':
                train_features, _ = model.encode(train_dataset)
                test_features, _ = model.encode(test_dataset)
            else:
                extractor = tf.keras.Sequential()
                for layer in model.layers[:-3]:
                    extractor.add(layer)
                extractor.built = True
                train_features = extractor.predict(train_dataset)
                test_features = extractor.predict(test_dataset)

            print(train_features.shape)
            print(test_features.shape)

            if part<=9:
                np.save('../../data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_0' + str(part), train_features)
                np.save('../../data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_0' + str(part), test_features)
            else:
                np.save('../../data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_' + str(part), train_features)
                np.save('../../data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_' + str(part), test_features)

            tf.keras.backend.clear_session()

    np.save('validation_losses_' + mode, validation_losses_mode)
