import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from itertools import combinations
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedShuffleSplit

from networks import *
from utils import *



os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.nice(10)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Global parameters

percentage_train = 80

num_crossval = 5
num_iterations = 5

modes = ['classall','classred','syllall','syllred','phonall','phonred','sound']

# Data parameters

frame_size = '1024'

# Network parameters

latent_dims = [16,32]

# Training parameters

epochs = 10000
batch_size = 128

# Class weighting

onset_loss_weight = 0.6
nucleus_loss_weight = 0.4
class_weight = {'onset': onset_loss_weight, 'nucleus': nucleus_loss_weight}

# Spectrogram normalisation values

norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

# Evaluation participants

list_test_participants_avp = [8,10,18,23]
list_test_participants_lvt = [0,6,7,13]

# Main loop

for m in range(len(modes)):

    mode = modes[m]

    if not os.path.isdir('models/' + mode):
        os.mkdir('models/' + mode)

    if not os.path.isdir('data/processed/' + mode):
        os.mkdir('data/processed/' + mode)

    if not os.path.isdir('data/processed/' + mode + '/reconstructions'):
        os.mkdir('data/processed/' + mode + '/reconstructions')

    for latent_dim in latent_dims:

        print('\n')
        print(mode)
        print('\n')

        # Spectrogram loading

        if mode=='sound':

            pretrain_dataset = np.zeros((1, 64, 48))
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_' + str(n).zfill(2)  + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_' + str(n).zfill(2)  + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

        else:

            pretrain_dataset = np.zeros((1, 64, 48))
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_Aug_' + str(n).zfill(2)  + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_' + str(n).zfill(2)  + '.npy')))
                    pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_Aug_' + str(n).zfill(2)  + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

        # Spectrogram normalisation

        pretrain_dataset = (pretrain_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
        pretrain_dataset = np.log(pretrain_dataset+1e-4)
        pretrain_dataset = (pretrain_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

        # Load and process classes

        if mode=='sound':
            classes = load_classes_sound(list_test_participants_avp, list_test_participants_lvt)
            num_classes = int(np.max(classes)+1)
        elif 'classall' in mode:
            load_classes_instrument(mode, list_test_participants_avp, list_test_participants_lvt)
            num_classes = np.max(classes)+1
        elif 'syll' in mode or 'phon' in mode:
            classes = load_classes_syll_phon(mode, list_test_participants_avp, list_test_participants_lvt)
            if 'syll' in mode:
                num_classes = np.max(classes)+1
            else:
                classes_onset = classes[0]
                classes_nucleus = classes[1]
                num_onset = np.max(classes_onset)+1
                num_nucleus = np.max(classes_nucleus)+1

        # Train models via 5-fold cross-validation (saving each model per fold)

        sss = StratifiedShuffleSplit(n_splits=num_crossval, test_size=0.2, random_state=0)

        if 'phon' in mode:
            pretrain_classes_split = classes_onset.copy()
        else:
            pretrain_classes_split = classes.copy()

        cv = 0

        for train_index, test_index in sss.split(pretrain_dataset, pretrain_classes_split):

            pretrain_dataset_train, pretrain_dataset_test = pretrain_dataset[train_index], pretrain_dataset[test_index]

            pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1).astype('float32')
            pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1).astype('float32')

            if 'phon' in mode:

                pretrain_classes_train_onset, pretrain_classes_test_onset = classes_onset[train_index], classes_onset[test_index]
                pretrain_classes_train_nucleus, pretrain_classes_test_nucleus = classes_nucleus[train_index], classes_nucleus[test_index]

                pretrain_classes_train_onset = pretrain_classes_train_onset.astype('float32')
                pretrain_classes_test_onset = pretrain_classes_test_onset.astype('float32')
                pretrain_classes_train_nucleus = pretrain_classes_train_nucleus.astype('float32')
                pretrain_classes_test_nucleus = pretrain_classes_test_nucleus.astype('float32')

            else:

                pretrain_classes_train, pretrain_classes_test = classes[train_index], classes[test_index]

                pretrain_classes_train = pretrain_classes_train.astype('float32')
                pretrain_classes_test = pretrain_classes_test.astype('float32')

            for it in range(num_iterations):

                patience_lr = 7
                patience_early = 14

                validation_accuracy = -1
                validation_loss = np.inf

                set_seeds(it)

                if 'phon' in mode:

                    set_seeds(it)

                    model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr=1e-3)

                    early_stopping = EarlyStopping_Phoneme(patience=patience_early, restore_best_weights=False)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_onset_accuracy', patience=patience_lr)

                    with tf.device(gpu_name):

                        history = model.fit(pretrain_dataset_train, [pretrain_classes_train_onset, pretrain_classes_train_nucleus], batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,[pretrain_classes_test_onset,pretrain_classes_test_nucleus]), callbacks=[early_stopping,lr_scheduler], class_weight=class_weight, shuffle=True)  # , verbose=0
                        validation_accuracy = (history.history['val_onset_accuracy'][-patience_early-1]+history.history['val_nucleus_accuracy'][-patience_early-1])/2
                        print(validation_accuracy)

                else:

                    if mode=='classall' or mode=='classred':
                        lr = 3*1e-4
                    elif mode=='syllall' or mode=='syllred':
                        lr = 5*1e-4
                    else:
                        lr = 5*1e-4
                        patience_early = 20
                        patience_lr = 10

                    model = CNN_Interim(num_classes, latent_dim)

                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early, restore_best_weights=False)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr)

                    with tf.device(gpu_name):

                        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                        history = model.fit(pretrain_dataset_train, pretrain_classes_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_classes_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0
                        validation_accuracy = max(history.history['val_accuracy'])
                        print(validation_accuracy)

                model.save_weights('models/' + mode + '/pretrained_' + mode + '_' + str(latent_dim) + '_' + str(cv) + '_' + str(it) + '.h5')

                # Compute processed features

                print('Computing features...')

                for part in list_test_participants_avp:

                    train_dataset = np.load('data/interim/AVP/Dataset_Train_' + str(part).zfill(2) + '.npy')
                    train_dataset_aug = np.load('data/interim/AVP/Dataset_Train_Aug_' + str(part).zfill(2) + '.npy')
                    test_dataset = np.load('data/interim/AVP/Dataset_Test_' + str(part).zfill(2) + '.npy')

                    train_dataset = (train_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    train_dataset = np.log(train_dataset+1e-4)
                    train_dataset = (train_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    train_dataset_aug = (train_dataset_aug-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    train_dataset_aug = np.log(train_dataset_aug+1e-4)
                    train_dataset_aug = (train_dataset_aug-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    test_dataset = (test_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    test_dataset = np.log(test_dataset+1e-4)
                    test_dataset = (test_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    train_dataset = np.expand_dims(train_dataset,axis=-1).astype('float32')
                    train_dataset_aug = np.expand_dims(train_dataset_aug,axis=-1).astype('float32')
                    test_dataset = np.expand_dims(test_dataset,axis=-1).astype('float32')

                    if 'phon' in mode:
                        extractor = tf.keras.Sequential()
                        for layer in model.layers[:-2]:
                            extractor.add(layer)
                        extractor.built = True
                        train_features = extractor.predict(train_dataset)
                        train_features_aug = extractor.predict(train_dataset_aug)
                        test_features = extractor.predict(test_dataset)
                    else:
                        extractor = tf.keras.Sequential()
                        for layer in model.cnn.layers[:-1]:
                            extractor.add(layer)
                        extractor.built = True
                        train_features = extractor.predict(train_dataset)
                        train_features_aug = extractor.predict(train_dataset_aug)
                        test_features = extractor.predict(test_dataset)

                    print(train_features.shape)
                    print(train_features_aug.shape)
                    print(test_features.shape)
                    print('')

                    np.save('data/processed/' + mode + '/train_features_avp_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), train_features)
                    np.save('data/processed/' + mode + '/train_features_aug_avp_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), train_features_aug)
                    np.save('data/processed/' + mode + '/test_features_avp_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), test_features) 

                for part in list_test_participants_lvt:

                    train_dataset = np.load('data/interim/LVT/Dataset_Train_' + str(part).zfill(2) + '.npy')
                    train_dataset_aug = np.load('data/interim/LVT/Dataset_Train_Aug_' + str(part).zfill(2) + '.npy')
                    test_dataset = np.load('data/interim/LVT/Dataset_Test_' + str(part).zfill(2) + '.npy')

                    train_dataset = (train_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    train_dataset = np.log(train_dataset+1e-4)
                    train_dataset = (train_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    train_dataset_aug = (train_dataset_aug-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    train_dataset_aug = np.log(train_dataset_aug+1e-4)
                    train_dataset_aug = (train_dataset_aug-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    test_dataset = (test_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                    test_dataset = np.log(test_dataset+1e-4)
                    test_dataset = (test_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                    train_dataset = np.expand_dims(train_dataset,axis=-1).astype('float32')
                    train_dataset_aug = np.expand_dims(train_dataset_aug,axis=-1).astype('float32')
                    test_dataset = np.expand_dims(test_dataset,axis=-1).astype('float32')

                    if 'phon' in mode:
                        extractor = tf.keras.Sequential()
                        for layer in model.layers[:-2]:
                            extractor.add(layer)
                        extractor.built = True
                        train_features = extractor.predict(train_dataset)
                        train_features_aug = extractor.predict(train_dataset_aug)
                        test_features = extractor.predict(test_dataset)
                    else:
                        extractor = tf.keras.Sequential()
                        for layer in model.cnn.layers[:-1]:
                            extractor.add(layer)
                        extractor.built = True
                        train_features = extractor.predict(train_dataset)
                        train_features_aug = extractor.predict(train_dataset_aug)
                        test_features = extractor.predict(test_dataset)

                    print(train_features.shape)
                    print(train_features_aug.shape)
                    print(test_features.shape)
                    print('')

                    np.save('data/processed/' + mode + '/train_features_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), train_features)
                    np.save('data/processed/' + mode + '/train_features_aug_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), train_features_aug)
                    np.save('data/processed/' + mode + '/test_features_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part).zfill(2) + '_' + str(cv) + '_' + str(it), test_features)

                tf.keras.backend.clear_session()

                cv += 1
