import os
import sys
import random
import numpy as np
import tensorflow as tf
from itertools import combinations
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
import utils
import attribution_methods

from networks import *



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

percentage_train = 80

num_crossval = 5
num_iterations = 5

modes = ['classall']

# Data parameters

frame_size = '1024'

# Network parameters

latent_dims = [32]

# Training parameters

epochs = 10000
batch_size = 128

# Class weighting

onset_loss_weight = 0.6
nucleus_loss_weight = 0.4
class_weight = {'onset': onset_loss_weight, 'nucleus': nucleus_loss_weight}

# Normalisation values

norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

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
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_' + str(n) + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

        else:

            pretrain_dataset = np.zeros((1, 64, 48))
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_Aug_' + str(n) + '.npy')))
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

        # Train models

        sss = StratifiedShuffleSplit(n_splits=num_crossval, test_size=0.2, random_state=0)

        if 'phon' in mode:
            pretrain_classes_split = classes_onset.copy()
        else:
            pretrain_classes_split = classes.copy()

        cv = 0

        # Placeholders

        grad_accum = np.zeros((5,5,30,4,64,48)) 
        spec_accum = np.zeros((30,64,48))

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

            if 'phon' in mode:
                model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr=5*1e-4)
            else:
                model = CNN_Interim(num_classes, latent_dim)

            model.built = True
            model.load_weights('models/' + mode + '/pretrained_' + mode + '_' + str(latent_dim) + '_' + str(cv) + '_' + str(it) + '.h5')

            part_indices = ['03','03','03','04','05','06','07','09','09','10','10','10','11','11','14',
                            '14','15','15','16','18','20','23','23','24','24','25','25','25','26','27']
            sound_indices = [101,48,80,25,23,23,85,73,102,13,38,63,13,63,
                            16,47,66,92,99,95,73,66,92,18,88,16,46,76,37,127]
            sound_labels = ['sd','hho','kd','hhc','hhc','hho','sd','kd','sd','hhc','hho','kd','hhc','kd','hhc',
                            'hho','kd','sd','sd','sd','kd','kd','sd','hhc','kd','hhc','hho','kd','hho','sd']

            norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]
            criterion = tf.keras.losses.SparseCategoricalCrossentropy()
            attrib = attribution_methods.AttributionMethods(0,90,attribution='abs_grad', replacement='spatial_analysis')

            for n in range(len(part_indices)):
                
                part_index = part_indices[n]
                sound_index = sound_indices[n]
                sound_label = sound_labels[n]

                # Spectrogram loading
                Spectrograms = np.load('data/interim/AVP/Dataset_Train_' + part_index + '.npy')
                Spectrogram = Spectrograms[sound_index]
                Label = sound_label
                
                if(Label=="kd"):
                    label_number=0
                elif(Label=="sd"):
                    label_number=1
                elif(Label=="hhc"):
                    label_number=2
                else:
                    label_number=3

                # Spectrogram normalisation
                Spectrogram = (Spectrogram-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                Spectrogram = np.log(Spectrogram+1e-4)
                Spectrogram = (Spectrogram-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)
                spec_accum[n,:,:] = Spectrogram
                ##### You can do your thing below
                for i in range(4):
                    grads = utils.saliency_map(model, Spectrogram[np.newaxis,:,:,np.newaxis], label=i)
                    inp, _ = attrib.apply_attribution(Spectrogram[np.newaxis,:,:], grads)
                    grad_accum[cv, it, n, i, :, :] = np.squeeze(inp)
                    grads = np.squeeze(grads)

            cv += 1

    hz = np.load('hz.npy')
    sec = np.load('sec.npy')

    for i in range(len(part_indices)):

        for j in range(4): 

            if j==0:
                predicted = "kd"
            elif j==1:
                predicted = "sd"
            elif j==2:
                predicted = "hhc"
            else:
                predicted = "hho"

            grads = np.squeeze(np.mean(grad_accum[:,:,i,j,:,:],axis=(0,1)))
            Spectrogram = spec_accum[i,:,:]

            plt.subplots(1,2,figsize=(7.5,4.5))
            plt.title(Label)
            plt.subplot(121)
            plt.imshow(Spectrogram, origin='lower')
            plt.title('Spectrogram', fontsize=20)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.ylabel('Frequency (Hz)', fontsize=14)

            plt.xticks(np.array([0, 11, 23, 35, 47]), np.array([0.,0.128, 0.267,0.406, 0.546]), fontsize=12)
            plt.yticks(np.array([0, 15, 31, 47, 63]), np.array([0, 952, 2713, 7735, 22050]), fontsize=12)
            
            plt.subplot(122)
            plt.imshow(grads)
            plt.title('Spatial saliency', fontsize=20)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.xticks(np.array([0, 11, 23, 35, 47]), np.array([0.,0.128, 0.267,0.406, 0.546]), fontsize=12)
            plt.yticks([])
                
            plt.gca().invert_yaxis()
            if sound_labels[i]==predicted:
                if i<=9:
                    plt.savefig('data/processed/spatial_abs_grad_bottom/saliency_0'+str(i)+'_'+sound_labels[i]+'_'+str(predicted)+'.png')
                else:
                    plt.savefig('data/processed/spatial_abs_grad_bottom/saliency_'+str(i)+'_'+sound_labels[i]+'_'+str(predicted)+'.png')
            plt.close()
