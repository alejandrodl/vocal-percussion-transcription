import os
import sys
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from itertools import combinations
import tensorflow_probability as tfp

sys.path.append('/homes/adl30/vocal-percussion-transcription')
from src.utils import *
from networks_offline import *



def triplet_loss(y_true, y_pred, margin=0.1):

    anchor_output, positive_output, negative_output = tf.split(y_pred, num_or_size_splits=3, axis=1)

    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0.0, margin + d_pos - d_neg)
    
    return tf.reduce_mean(loss)


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



os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
modes = ['triplet']
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

    for part in range(28):

        for a in range(len(frame_sizes)):

            frame_size = frame_sizes[a]

            print('\n')
            print([mode,frame_size,part])
            print('\n')

            # Spectrogram loading

            if mode!='triplet':

                # Create non-triplet dataset

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

            else:

                # Create triplet dataset

                print('Creating triplet dataset...')

                classes = np.zeros(1)

                classes_pre_triplet = np.zeros(1)
                pretriplet_dataset = np.zeros((1, 64, 64))
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            pretrain_data = np.load('../../data/interim/AVP/Dataset_Train_0' + str(n) + '_1024.npy')
                            classes_str = np.load('../../data/interim/AVP/Classes_Train_0' + str(n) + '.npy')
                            classes_pre = np.zeros(len(classes_str))
                            for nc in range(len(classes_str)):
                                if classes_str[nc]=='kd':
                                    classes_pre[nc] = (n*4)
                                elif classes_str[nc]=='sd':
                                    classes_pre[nc] = (n*4)+1
                                elif classes_str[nc]=='hhc':
                                    classes_pre[nc] = (n*4)+2
                                elif classes_str[nc]=='hho':
                                    classes_pre[nc] = (n*4)+3
                            classes_pre_triplet = np.concatenate((classes_pre_triplet, classes_pre))
                            pretriplet_dataset = np.vstack((pretriplet_dataset, pretrain_data))
                        else:
                            pretrain_data = np.load('../../data/interim/AVP/Dataset_Train_' + str(n) + '_1024.npy')
                            classes_str = np.load('../../data/interim/AVP/Classes_Train_' + str(n) + '.npy')
                            classes_pre = np.zeros(len(classes_str))
                            for nc in range(len(classes_str)):
                                if classes_str[nc]=='kd':
                                    classes_pre[nc] = (n*4)
                                elif classes_str[nc]=='sd':
                                    classes_pre[nc] = (n*4)+1
                                elif classes_str[nc]=='hhc':
                                    classes_pre[nc] = (n*4)+2
                                elif classes_str[nc]=='hho':
                                    classes_pre[nc] = (n*4)+3
                            classes_pre_triplet = np.concatenate((classes_pre_triplet, classes_pre))
                            pretriplet_dataset = np.vstack((pretriplet_dataset, pretrain_data))
                classes_pre_triplet = classes_pre_triplet[1:]
                pretriplet_dataset = pretriplet_dataset[1:]

                num_classes = int(np.max(classes_pre_triplet)+1)

                classes_pre_triplet_onset = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_pre_triplet_onset = np.concatenate((classes_pre_triplet_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_0' + str(n) + '.npy')))
                        else:
                            classes_pre_triplet_onset = np.concatenate((classes_pre_triplet_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_' + str(n) + '.npy')))
                classes_pre_triplet_onset = classes_pre_triplet_onset[1:]

                classes_pre_triplet_nucleus = np.zeros(1)
                for n in range(28):
                    if n==part:
                        continue
                    else:
                        if n<=9:
                            classes_pre_triplet_nucleus = np.concatenate((classes_pre_triplet_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                        else:
                            classes_pre_triplet_nucleus = np.concatenate((classes_pre_triplet_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_' + str(n) + '.npy')))
                classes_pre_triplet_nucleus = classes_pre_triplet_nucleus[1:]

                cmb = []
                classes_syllables = np.zeros(len(classes_pre_triplet_onset))
                for n in range(len(classes_pre_triplet_onset)):
                    combination = [classes_pre_triplet_onset[n],classes_pre_triplet_nucleus[n]]
                    if combination not in cmb:
                        cmb.append(combination)
                        classes_syllables[n] = cmb.index(combination)
                    else:
                        classes_syllables[n] = cmb.index(combination)

                class_start_part = int(part*4)
                class_end_part = class_start_part + 4

                counter = 0
                for n in range(num_classes):
                    class_start = n//4
                    class_end = class_start + 4
                    if class_start==class_start_part:
                        continue
                    else:
                        for it in range(2):
                            indices_sound = np.argwhere(classes_pre_triplet==n)[:,0]
                            syllables_sound = classes_syllables[indices_sound.tolist()]
                            unique_syllables = np.unique(syllables_sound)
                            for sy in range(len(unique_syllables)):
                                indices_sound_syll = indices_sound[np.argwhere(syllables_sound==unique_syllables[sy])[:,0].tolist()]
                                anchors_positives = list(combinations(indices_sound_syll.tolist(),2))
                                counter += len(anchors_positives)

                pretrain_dataset = np.zeros((counter,64,192))
                c = 0

                for n in range(num_classes):
                    class_start = n//4
                    class_end = class_start + 4
                    if class_start==class_start_part:
                        continue
                    else:
                        for it in range(2):
                            indices_sound = np.argwhere(classes_pre_triplet==n)[:,0]
                            syllables_sound = classes_syllables[indices_sound.tolist()]
                            unique_syllables = np.unique(syllables_sound)
                            indices_sounds_part = list(set(np.argwhere(classes_pre_triplet>=class_start)[:,0])&set(np.argwhere(classes_pre_triplet<class_end)[:,0]))
                            indices_sounds_other = np.concatenate((np.argwhere(classes_pre_triplet<class_start)[:,0],np.argwhere(classes_pre_triplet>=class_end)[:,0]))
                            for sy in range(len(unique_syllables)):
                                indices_sound_syll = indices_sound[np.argwhere(syllables_sound==unique_syllables[sy])[:,0].tolist()]
                                anchors_positives = list(combinations(indices_sound_syll.tolist(),2))
                                L = len(anchors_positives)
                                L_half = L//2
                                negatives_part = np.random.choice(indices_sounds_part, size=L_half, replace=True)
                                negatives_other = np.random.choice(indices_sounds_other, size=L-L_half, replace=True)
                                for s in range(L_half):
                                    pretrain_dataset[c,:,:64] = pretriplet_dataset[anchors_positives[s][0]]
                                    pretrain_dataset[c,:,64:128] = pretriplet_dataset[anchors_positives[s][1]]
                                    pretrain_dataset[c,:,128:192] = pretriplet_dataset[negatives_part[s]]
                                    c += 1
                                for s in range(L_half,L):
                                    pretrain_dataset[c,:,:64] = pretriplet_dataset[anchors_positives[s][0]]
                                    pretrain_dataset[c,:,64:128] = pretriplet_dataset[anchors_positives[s][1]]
                                    pretrain_dataset[c,:,128:192] = pretriplet_dataset[negatives_other[s-L_half]]
                                    c += 1

                print('Done.')

            # Spectrogram normalisation

            pretrain_dataset = (pretrain_dataset-norm_min_max_1[1][0])/(norm_min_max_1[1][1]-norm_min_max_1[1][0]+1e-16)
            pretrain_dataset = np.log(pretrain_dataset+1e-4)
            pretrain_dataset = (pretrain_dataset-norm_min_max_2[1][0])/(norm_min_max_2[1][1]-norm_min_max_2[1][0]+1e-16)
            
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

            # Further class processing

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

            elif mode=='triplet':

                pretrain_classes_train = np.zeros(pretrain_dataset_train.shape[0]).astype('float32')
                pretrain_classes_test = np.zeros(pretrain_dataset_test.shape[0]).astype('float32')

            # Train models

            if mode=='vae':

                validation_losses_mode[a,part] = 1

                while validation_losses_mode[a,part] > 0.003:

                    model = VAE_Interim(latent_dim)

                    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

                    with tf.device(gpu_name):

                        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=False)
                        history = model.fit(pretrain_dataset_train, pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_dataset_test), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)  # , verbose=0
                        validation_losses_mode[a,part] = min(history.history['val_loss'])
                        print(validation_losses_mode[a,part])

            elif 'phon' in mode:

                while validation_losses_mode[a,part] < min_acc[m-1]:

                    model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr=3*1e-4)

                    early_stopping = EarlyStopping_Phoneme(patience=7)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_onset_accuracy', patience=14)

                    with tf.device(gpu_name):

                        history = model.fit(pretrain_dataset_train, [pretrain_classes_train_onset, pretrain_classes_train_nucleus], batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,[pretrain_classes_test_onset,pretrain_classes_test_nucleus]), callbacks=[early_stopping,lr_scheduler], class_weight=class_weight, shuffle=True, verbose=0)  # , verbose=0
                        validation_losses_mode[a,part] = (history.history['val_onset_accuracy'][-patience_early-1]+history.history['val_nucleus_accuracy'][-patience_early-1])/2
                        print(validation_losses_mode[a,part])

            elif 'triplet' in mode:

                validation_losses_mode[a,part] = 1

                while validation_losses_mode[a,part] > 0.006:

                    model = CNN_Interim_Triplet(latent_dim)

                    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)

                    with tf.device(gpu_name):

                        model.compile(optimizer=optimizer, loss=triplet_loss)
                        history = model.fit(pretrain_dataset_train, pretrain_classes_train, batch_size=1024, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_classes_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0
                        validation_losses_mode[a,part] = min(history.history['val_loss'])
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
                        validation_losses_mode[a,part] = max(history.history['val_accuracy'])
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

            train_dataset = (train_dataset-norm_min_max_1[1][0])/(norm_min_max_1[1][1]-norm_min_max_1[1][0]+1e-16)
            train_dataset = np.log(train_dataset+1e-4)
            train_dataset = (train_dataset-norm_min_max_2[1][0])/(norm_min_max_2[1][1]-norm_min_max_2[1][0]+1e-16)

            test_dataset = (test_dataset-norm_min_max_1[1][0])/(norm_min_max_1[1][1]-norm_min_max_1[1][0]+1e-16)
            test_dataset = np.log(test_dataset+1e-4)
            test_dataset = (test_dataset-norm_min_max_2[1][0])/(norm_min_max_2[1][1]-norm_min_max_2[1][0]+1e-16)

            train_dataset = np.expand_dims(train_dataset,axis=-1).astype('float32')
            test_dataset = np.expand_dims(test_dataset,axis=-1).astype('float32')

            if 'phon' in mode:
                extractor = tf.keras.Sequential()
                for layer in model.layers[:-2]:
                    extractor.add(layer)
                extractor.built = True
                train_features = extractor.predict(train_dataset)
                test_features = extractor.predict(test_dataset)
            elif mode=='vae':
                train_features, _ = model.encode(train_dataset)
                test_features, _ = model.encode(test_dataset)
            elif mode=='triplet':
                train_features = model.cnn(train_dataset)
                test_features = model.cnn(test_dataset)
            else:
                extractor = tf.keras.Sequential()
                for layer in model.cnn.layers[:-1]:
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
