import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from networks_offline import *
from src.utils import *



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

percentage_train = 85

epochs = 10000
patience_lr = 5
patience_early = 10

batch_size = 1024
num_examples_to_generate = 16

dropout = 0.5
num_filters = 64
num_filters_str = '64'
latent_dim = 32

frame_sizes = ['512','1024','2048']
lrs = 1e-3

mode = ['vae','classall','classred','syllall','syllred','phonall','phonred'] # Triplet!!

# AVP Personal

for frame_size in frame_sizes:

    for part in range(28):

        # Load and process spectrograms

        pretrain_dataset = np.zeros((1, 64, 64))
        for n in range(28):
            if n==part:
                continue
            else:
                if n<=9:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/external/AVP/Dataset_Train_Aug_0' + str(n) + '_' + frame_size + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/external/AVP/Dataset_Test_Aug_0' + str(n) + '_' + frame_size + '.npy')))
                else:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/external/AVP/Dataset_Train_Aug_' + str(n) + '_' + frame_size + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/external/AVP/Dataset_Test_Aug_' + str(n) + '_' + frame_size + '.npy')))

        pretrain_dataset = pretrain_dataset[1:]
        print(pretrain_dataset.shape)

        pretrain_dataset = (pretrain_dataset-np.min(pretrain_dataset))/(np.max(pretrain_dataset)-np.min(pretrain_dataset))
        print([np.min(pretrain_dataset),np.max(pretrain_dataset)])
        pretrain_dataset = np.log(pretrain_dataset+1e-4)
        pretrain_dataset = (pretrain_dataset-np.min(pretrain_dataset))/(np.max(pretrain_dataset)-np.min(pretrain_dataset))
        print([np.min(pretrain_dataset),np.max(pretrain_dataset)])

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
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
            classes_onset = classes_onset[1:]

            classes_nucleus = np.zeros(1)
            for n in range(28):
                if n==part:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
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
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('../../data/external/AVP/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
            classes_onset = classes_onset[1:]

            classes_nucleus = np.zeros(1)
            for n in range(28):
                if n==part:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/external/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
            classes_nucleus = classes_nucleus[1:]

            num_onset = np.max(classes_onset)+1
            num_nucleus = np.max(classes_nucleus)+1

        elif 'classall' in mode:

            classes_str = np.zeros(1)
            for n in range(28):
                if n>=part_start and n<part_end:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
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

        elif 'classred' in mode:

            classes_str = np.zeros(1)
            for n in range(28):
                if n>=part_start and n<part_end:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('../../data/external/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
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

            np.random.seed(0)
            np.random.shuffle(classes)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test)
            
            train_classes = classes[:cutoff_train].astype('float32')
            test_classes = classes[cutoff_train:].astype('float32')

        elif 'phon' in mode:

            np.random.seed(0)
            np.random.shuffle(pretrain_classes_train)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test)
            
            train_classes_onset = Classes_Onset[:cutoff_train].astype('float32')
            train_classes_nucleus = Classes_Nucleus[:cutoff_train].astype('float32')
            test_classes_onset = Classes_Onset[cutoff_train:].astype('float32')
            test_classes_nucleus = Classes_Nucleus[cutoff_train:].astype('float32')

        elif 'class' in mode:

            np.random.seed(0)
            np.random.shuffle(classes)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test)
            
            train_classes = classes[:cutoff_train].astype('float32')
            test_classes = classes[cutoff_train:].astype('float32')

        # Train models

        for it in range(10):

            if mode=='vae':

                model = VAE_Interim(latent_dim)

                optimizer = tf.keras.optimizers.Adam(lr)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

                with tf.device(gpu_name):

                    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=False)
                    history = model.fit(pretrain_dataset_train, pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_dataset_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0

                    fpath = '../../models/pretrained_' + mode + '_' + frame_size + '_' + str(it)
                    model.save_weights(fpath+'.h5')

            elif 'phon' in mode:

                model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr)

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

                with tf.device(gpu_name):

                    history = model.fit(pretrain_dataset_train, [train_classes_onset, train_classes_nucleus], batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,[test_classes_onset,test_classes_nucleus]), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0

                    fpath = '../../models/pretrained_' + mode + '_' + frame_size + '_' + str(it)
                    model.save_weights(fpath+'.h5')

            else:

                model = CNN_Interim(latent_dim)

                optimizer = tf.keras.optimizers.Adam(lr)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr)

                with tf.device(gpu_name):

                    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                    history = model.fit(pretrain_dataset_train, pretrain_classes_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_classes_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0

                    fpath = '../../models/pretrained_' + mode + '_' + frame_size + '_' + str(it)
                    model.save_weights(fpath+'.h5')
