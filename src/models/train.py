import os
import sys
import random
import numpy as np
import tensorflow as tf
from itertools import combinations
import tensorflow_probability as tfp

from networks import *



# Mode parameters

modes = ['classall','classred','syllall','syllred','phonall','phonred']

# Global parameters

percentage_train = 80

num_crossval = 5
num_iterations = 5

# Data parameters

frame_size = '1024'
latent_dim = 32

##### This is the main loop with the "modes", "num_crossval", and "num_iterations" parameters. From here to the next
# stop point in line 322 it's all about loading the classes vectors and counting the number of unique classes, in
# order to load the models, so don't worry about the code in-between (the classes vectors vary between loop iterations).

for m in range(len(modes)):

    mode = modes[m]

    print('\n')
    print(mode)
    print('\n')

    list_test_participants_avp = [8,10,18,23]
    list_test_participants_lvt = [0,6,7,13]

    num_samples = 52065
    cutoff_test = int(((100-percentage_train)/100)*num_samples)
    
    for cv in range(num_crossval):

        idx_start = cv*cutoff_test
        idx_end = (cv+1)*cutoff_test
        idxs_test = np.arange(idx_start,idx_end).tolist()

        # Load and process classes

        if 'syllall' in mode or 'phonall' in mode:

            classes_onset = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
            classes_onset = classes_onset[1:]

            classes_nucleus = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
            classes_nucleus = classes_nucleus[1:]

            train_classes_onset = np.delete(classes_onset,idxs_test,axis=0).astype('float32')
            test_classes_onset = classes_onset[idx_start:idx_end].astype('float32')

            train_classes_nucleus = np.delete(classes_nucleus,idxs_test,axis=0).astype('float32')
            test_classes_nucleus = classes_nucleus[idx_start:idx_end].astype('float32')

            num_onset = np.max(classes_onset)+1
            num_nucleus = np.max(classes_nucleus)+1

        elif 'syllred' in mode or 'phonred' in mode:

            classes_onset = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
            classes_onset = classes_onset[1:]

            classes_nucleus = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                        classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
            classes_nucleus = classes_nucleus[1:]

            num_onset = np.max(classes_onset)+1
            num_nucleus = np.max(classes_nucleus)+1

        elif 'classall' in mode:

            classes_str = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
            classes_str = classes_str[1:]

            classes = np.zeros(len(classes_str))
            for n in range(len(classes_str)):
                if classes_str[n]=='kd' or classes_str[n]=='Kick':
                    classes[n] = 0
                elif classes_str[n]=='sd' or classes_str[n]=='Snare':
                    classes[n] = 1
                elif classes_str[n]=='hhc' or classes_str[n]=='HH':
                    classes[n] = 2
                elif classes_str[n]=='hho':
                    classes[n] = 3

            num_classes = np.max(classes)+1

        elif 'classred' in mode:

            classes_str = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                        classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
            classes_str = classes_str[1:]

            classes = np.zeros(len(classes_str))
            for n in range(len(classes_str)):
                if classes_str[n]=='kd' or classes_str[n]=='Kick':
                    classes[n] = 0
                elif classes_str[n]=='sd' or classes_str[n]=='Snare':
                    classes[n] = 0
                elif classes_str[n]=='hhc' or classes_str[n]=='HH':
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
            
            pretrain_classes_train = np.delete(classes,idxs_test,axis=0).astype('float32')
            pretrain_classes_test = classes[idx_start:idx_end].astype('float32')

            np.random.seed(0)
            np.random.shuffle(pretrain_classes_train)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test)

        elif 'phon' in mode:

            pretrain_classes_train_onset = np.delete(classes_onset,idxs_test,axis=0).astype('float32')
            pretrain_classes_test_onset = classes_onset[idx_start:idx_end].astype('float32')

            pretrain_classes_train_nucleus = np.delete(classes_nucleus,idxs_test,axis=0).astype('float32')
            pretrain_classes_test_nucleus = classes_nucleus[idx_start:idx_end].astype('float32')

            np.random.seed(0)
            np.random.shuffle(pretrain_classes_train_onset)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_train_nucleus)

            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test_onset)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test_nucleus)

        elif 'class' in mode or 'siamese' in mode:

            pretrain_classes_train = np.delete(classes,idxs_test,axis=0).astype('float32')
            pretrain_classes_test = classes[idx_start:idx_end].astype('float32')

            np.random.seed(0)
            np.random.shuffle(pretrain_classes_train)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_classes_test)

        # Train models for 5 iterations

        for it in range(num_iterations):

            ##### By now, you have the num_classes parameter of the iteration. If the current mode is phonall or phonred,
            # you'll have num_onset and num_nucleus, as it's multi-task learning classification.

            if 'phon' in mode:
                model = CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr=5*1e-4)
            else:
                model = CNN_Interim(num_classes, latent_dim)

            model.built = True
            model.load_weights('models/' + mode + '/pretrained_' + mode + '_' + str(latent_dim) + '_' + str(cv) + '_' + str(it) + '.h5')

            # Compute processed features

            part_indices = ['03','03','03','04','05','06','07','09','09','10','10','10','11','11','14',
                            '14','15','15','16','18','20','23','23','24','24','25','25','25','26','27']
            sound_indices = [101,48,80,25,23,23,85,73,102,13,38,63,13,63,
                            16,47,66,92,99,95,73,66,92,18,88,16,46,76,37,127]
            sound_labels = ['sd','hho','kd','hhc','hhc','hho','sd','kd','sd','hhc','hho','kd','hhc','kd','hhc',
                            'hho','kd','sd','sd','sd','kd','kd','sd','hhc','kd','hhc','hho','kd','hho','sd']

            norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

            for n in range(len(part_indices)):
                
                part_index = part_indices[n]
                sound_index = sound_indices[n]
                sound_label = sound_labels[n]

                # Spectrogram loading

                Spectrograms = np.load('data/interim/AVP/Dataset_Train_' + part_index + '.npy')
                Spectrogram = Spectrograms[sound_index]
                Label = sound_label

                # Spectrogram normalisation

                Spectrogram = (Spectrogram-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
                Spectrogram = np.log(Spectrogram+1e-4)
                Spectrogram = (Spectrogram-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

                ##### You can do your thing below.


