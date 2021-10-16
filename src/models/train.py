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

from networks_2 import *




class EarlyStopping_Phoneme(tf.keras.callbacks.Callback):
    '''
    Function for early stopping for phoneme labels. It considers both the onset and nucleus losses.
    '''
    def __init__(self, patience=0, restore_best_weights=False):
        super(EarlyStopping_Phoneme, self).__init__()

        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = 0

    def on_epoch_end(self, epoch, logs=None):

        onset_loss = logs.get('val_onset_accuracy')
        nucleus_loss = logs.get('val_nucleus_accuracy')

        if np.greater(onset_loss+nucleus_loss, self.best_loss):
            self.best_loss = onset_loss+nucleus_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)



os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

#modes = ['vae','classall','classred','syllall','syllred','phonall','phonred','sound']
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
        #pretrain_dataset = np.log(pretrain_dataset+1e-4)
        #pretrain_dataset = (pretrain_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

        # Load and process classes

        if mode=='vae':

            classes = np.zeros(pretrain_dataset.shape[0])

        elif mode=='sound':

            classes = np.zeros(1)
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        classes_str = np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')
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
                        classes = np.concatenate((classes, classes_pre))
                        classes_str = np.load('data/interim/AVP/Classes_Test_0' + str(n) + '.npy')
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
                        classes = np.concatenate((classes, classes_pre))
                    else:
                        classes_str = np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')
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
                        classes = np.concatenate((classes, classes_pre))
                        classes_str = np.load('data/interim/AVP/Classes_Test_' + str(n) + '.npy')
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
                        classes = np.concatenate((classes, classes_pre))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        classes_str = np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')
                        classes_pre = np.zeros(len(classes_str))
                        for nc in range(len(classes_str)):
                            if classes_str[nc]=='Kick':
                                classes_pre[nc] = (28*4) + (n*3)
                            elif classes_str[nc]=='Snare':
                                classes_pre[nc] = (28*4) + (n*3)+1
                            elif classes_str[nc]=='HH':
                                classes_pre[nc] = (28*4) + (n*3)+2
                        classes = np.concatenate((classes, classes_pre))
                        classes_str = np.load('data/interim/LVT/Classes_Test_0' + str(n) + '.npy')
                        classes_pre = np.zeros(len(classes_str))
                        for nc in range(len(classes_str)):
                            if classes_str[nc]=='Kick':
                                classes_pre[nc] = (28*4) + (n*3)
                            elif classes_str[nc]=='Snare':
                                classes_pre[nc] = (28*4) + (n*3)+1
                            elif classes_str[nc]=='HH':
                                classes_pre[nc] = (28*4) + (n*3)+2
                        classes = np.concatenate((classes, classes_pre))
                    else:
                        classes_str = np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')
                        classes_pre = np.zeros(len(classes_str))
                        for nc in range(len(classes_str)):
                            if classes_str[nc]=='Kick':
                                classes_pre[nc] = (28*4) + (n*3)
                            elif classes_str[nc]=='Snare':
                                classes_pre[nc] = (28*4) + (n*3)+1
                            elif classes_str[nc]=='HH':
                                classes_pre[nc] = (28*4) + (n*3)+2
                        classes = np.concatenate((classes, classes_pre))
                        classes_str = np.load('data/interim/LVT/Classes_Test_' + str(n) + '.npy')
                        classes_pre = np.zeros(len(classes_str))
                        for nc in range(len(classes_str)):
                            if classes_str[nc]=='Kick':
                                classes_pre[nc] = (28*4) + (n*3)
                            elif classes_str[nc]=='Snare':
                                classes_pre[nc] = (28*4) + (n*3)+1
                            elif classes_str[nc]=='HH':
                                classes_pre[nc] = (28*4) + (n*3)+2
                        classes = np.concatenate((classes, classes_pre))
            classes = classes[1:]

            num_classes = int(np.max(classes)+1)

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

            num_onset = np.max(classes_onset)+1
            num_nucleus = np.max(classes_nucleus)+1

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

        # Train models

        sss = StratifiedShuffleSplit(n_splits=num_crossval, test_size=0.2, random_state=0)

        if 'phon' in mode:
            pretrain_classes_split = classes_onset.copy()
        else:
            pretrain_classes_split = classes.copy()

        cv = 0

        grad_accum = np.zeros((5,5,30,4,64,48)) 
        spec_accum = np.zeros((30,64,48))

        for train_index, test_index in sss.split(pretrain_dataset, pretrain_classes_split):

            pretrain_dataset_train, pretrain_dataset_test = pretrain_dataset[train_index], pretrain_dataset[test_index]
            
            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_train)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_test)

            pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1).astype('float32')
            pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1).astype('float32')

            if 'phon' in mode:

                pretrain_classes_train_onset, pretrain_classes_test_onset = classes_onset[train_index], classes_onset[test_index]
                pretrain_classes_train_nucleus, pretrain_classes_test_nucleus = classes_nucleus[train_index], classes_nucleus[test_index]

                pretrain_classes_train_onset = pretrain_classes_train_onset.astype('float32')
                pretrain_classes_test_onset = pretrain_classes_test_onset.astype('float32')
                pretrain_classes_train_nucleus = pretrain_classes_train_nucleus.astype('float32')
                pretrain_classes_test_nucleus = pretrain_classes_test_nucleus.astype('float32')

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train_onset)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train_nucleus)

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test_onset)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test_nucleus)

            else:

                pretrain_classes_train, pretrain_classes_test = classes[train_index], classes[test_index]

                pretrain_classes_train = pretrain_classes_train.astype('float32')
                pretrain_classes_test = pretrain_classes_test.astype('float32')

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
                    #grads = utils.loss_gradient(model, Spectrogram[np.newaxis,:,:,np.newaxis], label=i, loss_function=criterion)
                    inp, _ = attrib.apply_attribution(Spectrogram[np.newaxis,:,:], grads)
                    grad_accum[cv, it, n, i, :, :] = np.squeeze(inp)
                    grads = np.squeeze(grads)
                    #grads =grads/np.sqrt(np.mean(grads**2))
                #ax[0,0].invert_yaxis()
                    """ 
                    
                    plt.subplots(2,2, sharex=True, sharey=True)
                    plt.title(Label)
                    plt.subplot(221)
                    plt.imshow(Spectrogram)
                    plt.title('Spectrogram')
                    plt.xlabel('Time stamps')
                    plt.ylabel('Frequency')
                
                    plt.subplot(222)
                    plt.imshow(grads)
                    plt.title('Raw saliency')
                    plt.xlabel('Time stamps')
                    plt.ylabel('Frequency')
                
                    plt.subplot(223)
                    plt.imshow(np.abs(grads))
                    plt.title('Absolute saliency')
                    plt.xlabel('Time stamps')
                    plt.ylabel('Frequency')
                 
                    plt.subplot(224)
                    plt.imshow(np.multiply(grads, Spectrogram))
                    plt.title('saliency x input')
                    plt.xlabel('Time stamps')
                    plt.ylabel('Frequency')
                
                    plt.gca().invert_yaxis()
                    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
                    cax = plt.axes([0.85, 0.1, 0.02, 0.8])
                    plt.colorbar(cax=cax)
                    plt.savefig('loss/'+mode+'_'+str(latent_dim)+'_'+str(cv)+'_'+str(it)+'_part_indices_'+str(n)+'_gt_'+Label+'_predicted_'+str(i)+'.png')
                    plt.close()
                    """
            #input('wait')

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

            #grads = np.squeeze(grad_accum[0,0,i,j,:,:])
            #grads = np.flip(np.squeeze(np.mean(grad_accum[:,:,i,j,:,:],axis=(0,1))), axis=0)
            grads = np.squeeze(np.mean(grad_accum[:,:,i,j,:,:],axis=(0,1)))
            #grads = np.clip(grads, np.mean(grads)-3*np.std(grads), np.mean(grads)+3*np.std(grads))
            #Spectrogram = np.flip(spec_accum[i,:,:], axis=0)
            Spectrogram = spec_accum[i,:,:]
            plt.subplots(1,2,figsize=(7.5,4.5))
            plt.title(Label)
            plt.subplot(121)
            #plt.imshow(Spectrogram)
            plt.imshow(Spectrogram, origin='lower')
            plt.title('Spectrogram', fontsize=20)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.ylabel('Frequency (Hz)', fontsize=14)

            plt.xticks(np.array([0, 11, 23, 35, 47]), np.array([0.,0.128, 0.267,0.406, 0.546]), fontsize=12)
            plt.yticks(np.array([0, 15, 31, 47, 63]), np.array([0, 952, 2713, 7735, 22050]), fontsize=12)

            #plt.xticklabels(sec)
            #plt.yticklabels(hz)
            
            plt.subplot(122)
            plt.imshow(grads)
            plt.title('Spatial saliency', fontsize=20)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.xticks(np.array([0, 11, 23, 35, 47]), np.array([0.,0.128, 0.267,0.406, 0.546]), fontsize=12)
            plt.yticks([])
            #plt.xticks(np.array([0, 11, 23, 35, 47]), np.array([0.,0.128, 0.267,0.406, 0.546]))
            #plt.yticks(np.array([0, 15, 31, 47, 63]), np.array([0, 952, 2713, 7735, 22050]))
            
            #plt.xticks(np.array([0, 11, 23, 35, 47]), sec[np.array([0,11,23,35,47]).round(3)])
            #plt.yticks(np.array([0, 15, 31, 47, 63]), hz[np.array([0, 15,31,47,63]).round(3)]) 
            print(sec[np.array([0,11,23,35,47]).round(3)])
            print(hz[np.array([0,15,31,47,63])]) 

            #plt.set_xticklabels(sec)
            #plt.set_yticklabels(hz)   
            #plt.subplot(223)
            #plt.imshow(np.abs(grads))
            #plt.title('Absolute saliency')
            #plt.xlabel('Time stamps')
            #plt.ylabel('Frequency')
              
            #plt.subplot(224)
            #plt.imshow(np.multiply(grads, Spectrogram))
            #plt.title('saliency x input')
            #plt.xlabel('Time stamps')
            #plt.ylabel('Frequency')
                
            plt.gca().invert_yaxis()
            #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            #cax = plt.axes([0.85, 0.1, 0.02, 0.8])
            #plt.colorbar(cax=cax)
            if sound_labels[i]==predicted:
                if i<=9:
                    plt.savefig('src/models/spatial_abs_grad_bottom/saliency_0'+str(i)+'_'+sound_labels[i]+'_'+str(predicted)+'.png')
                else:
                    plt.savefig('src/models/spatial_abs_grad_bottom/saliency_'+str(i)+'_'+sound_labels[i]+'_'+str(predicted)+'.png')
            plt.close()
