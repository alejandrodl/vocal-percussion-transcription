import os
import sys
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from itertools import combinations
import tensorflow_probability as tfp

from networks_offline import *




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

        if np.greater(0.5*onset_loss+0.5*nucleus_loss, self.best_loss):
            self.best_loss = 0.5*onset_loss+0.5*nucleus_loss
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

modes = ['vae','classall','classred','syllall','syllred','phonall','phonred','sound']
min_acc = [0.10,0.15,0.10,0.15,0.10,0.15,0.0]

# Data parameters

frame_size = '1024'

# Network parameters

latent_dims = [16,32]

# Training parameters

epochs = 10000
batch_size = 512

# Class weighting

onset_loss_weight = 0.5
nucleus_loss_weight = 0.5
class_weight = {'onset': onset_loss_weight, 'nucleus': nucleus_loss_weight}

# Normalisation values

norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

list_test_participants_avp = [8,10,18,23]
list_test_participants_lvt = [0,6,7,13]

# Main loop

for m in range(len(modes)):

    mode = modes[m]

    if not os.path.isdir('../../models/' + mode):
        os.mkdir('../../models/' + mode)

    if not os.path.isdir('../../data/processed/' + mode):
        os.mkdir('../../data/processed/' + mode)

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
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Test_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Test_' + str(n) + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

        else:

            pretrain_dataset = np.zeros((1, 64, 48))
            for n in range(28):
                if n in list_test_participants_avp:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_' + str(n) + '.npy')))
            for n in range(20):
                if n in list_test_participants_lvt:
                    continue
                else:
                    if n<=9:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_Aug_0' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Test_Aug_0' + str(n) + '.npy')))
                    else:
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Train_Aug_' + str(n) + '.npy')))
                        pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/LVT/Dataset_Test_Aug_' + str(n) + '.npy')))
            pretrain_dataset = pretrain_dataset[1:]

        # Spectrogram normalisation

        pretrain_dataset = (pretrain_dataset-norm_min_max[0][0])/(norm_min_max[0][1]-norm_min_max[0][0]+1e-16)
        pretrain_dataset = np.log(pretrain_dataset+1e-4)
        pretrain_dataset = (pretrain_dataset-norm_min_max[1][0])/(norm_min_max[1][1]-norm_min_max[1][0]+1e-16)

        # Cross-Validation

        cutoff_test = int(((100-percentage_train)/100)*pretrain_dataset.shape[0])
        
        for cv in range(num_crossval):

            idx_start = cv*cutoff_test
            idx_end = (cv+1)*cutoff_test
            idxs_test = np.arange(idx_start,idx_end).tolist()

            pretrain_dataset_cv = pretrain_dataset.copy()
            pretrain_dataset_train = np.delete(pretrain_dataset_cv,idxs_test,axis=0).astype('float32')
            pretrain_dataset_test = pretrain_dataset_cv[idx_start:idx_end].astype('float32')

            pretrain_dataset_train = np.expand_dims(pretrain_dataset_train,axis=-1).astype('float32')
            pretrain_dataset_test = np.expand_dims(pretrain_dataset_test,axis=-1).astype('float32')

            print(pretrain_dataset_train.shape)
            print(pretrain_dataset_test.shape)
            
            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_train)

            np.random.seed(0)
            np.random.shuffle(pretrain_dataset_test)

            # Load and process classes

            if mode=='sound':

                classes = np.zeros(1)
                for n in range(28):
                    if n in list_test_participants_avp:
                        continue
                    else:
                        if n<=9:
                            classes_str = np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')
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
                            classes_str = np.load('../../data/interim/AVP/Classes_Test_0' + str(n) + '.npy')
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
                            classes_str = np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')
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
                            classes_str = np.load('../../data/interim/AVP/Classes_Test_' + str(n) + '.npy')
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
                            classes_str = np.load('../../data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')
                            classes_pre = np.zeros(len(classes_str))
                            for nc in range(len(classes_str)):
                                if classes_str[nc]=='Kick':
                                    classes_pre[nc] = (28*4) + (n*3)
                                elif classes_str[nc]=='Snare':
                                    classes_pre[nc] = (28*4) + (n*3)+1
                                elif classes_str[nc]=='HH':
                                    classes_pre[nc] = (28*4) + (n*3)+2
                            classes = np.concatenate((classes, classes_pre))
                            classes_str = np.load('../../data/interim/LVT/Classes_Test_0' + str(n) + '.npy')
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
                            classes_str = np.load('../../data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')
                            classes_pre = np.zeros(len(classes_str))
                            for nc in range(len(classes_str)):
                                if classes_str[nc]=='Kick':
                                    classes_pre[nc] = (28*4) + (n*3)
                                elif classes_str[nc]=='Snare':
                                    classes_pre[nc] = (28*4) + (n*3)+1
                                elif classes_str[nc]=='HH':
                                    classes_pre[nc] = (28*4) + (n*3)+2
                            classes = np.concatenate((classes, classes_pre))
                            classes_str = np.load('../../data/interim/LVT/Classes_Test_' + str(n) + '.npy')
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
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Train_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Train_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
                classes_onset = classes_onset[1:]

                classes_nucleus = np.zeros(1)
                for n in range(28):
                    if n in list_test_participants_avp:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Train_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
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
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/AVP/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_onset = np.concatenate((classes_onset, np.load('../../data/interim/LVT/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
                classes_onset = classes_onset[1:]

                classes_nucleus = np.zeros(1)
                for n in range(28):
                    if n in list_test_participants_avp:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                            classes_nucleus = np.concatenate((classes_nucleus, np.load('../../data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
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
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
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
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
                for n in range(20):
                    if n in list_test_participants_lvt:
                        continue
                    else:
                        if n<=9:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                            classes_str = np.concatenate((classes_str, np.load('../../data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
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

            elif 'class' in mode or 'sound' in mode:

                pretrain_classes_train = np.delete(classes,idxs_test,axis=0).astype('float32')
                pretrain_classes_test = classes[idx_start:idx_end].astype('float32')

                np.random.seed(0)
                np.random.shuffle(pretrain_classes_train)
                
                np.random.seed(0)
                np.random.shuffle(pretrain_classes_test)

            # Train models

            for it in range(num_iterations):

                patience_lr = 7
                patience_early = 14

                validation_accuracy = -1
                validation_loss = np.inf

                set_seeds(it)

                if mode=='vae':

                    while validation_loss > 0.003:

                        set_seeds(it)

                        model = VAE_Interim(latent_dim)

                        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early, restore_best_weights=False)
                        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

                        with tf.device(gpu_name):

                            model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=False)
                            history = model.fit(pretrain_dataset_train, pretrain_dataset_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_dataset_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0
                            validation_loss = min(history.history['val_loss'])
                            print(validation_loss)

                elif 'phon' in mode:

                    while validation_accuracy < min_acc[m-1]:

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

                    while validation_accuracy < min_acc[m-1]:

                        set_seeds(it)

                        model = CNN_Interim(num_classes, latent_dim)

                        optimizer = tf.keras.optimizers.Adam(lr=lr)
                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early, restore_best_weights=False)
                        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr)

                        with tf.device(gpu_name):

                            model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                            history = model.fit(pretrain_dataset_train, pretrain_classes_train, batch_size=batch_size, epochs=epochs, validation_data=(pretrain_dataset_test,pretrain_classes_test), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0
                            validation_accuracy = max(history.history['val_accuracy'])
                            print(validation_accuracy)

                model.save_weights('../../models/' + mode + '/pretrained_' + mode + '_' + str(latent_dim) + '_' + str(cv) + '_' + str(it) + '.h5')

                # Compute processed features

                print('Computing features...')

                for part in list_test_participants_avp:

                    if part<=9:
                        train_dataset = np.load('../../data/interim/AVP/Dataset_Train_0' + str(part) + '.npy')
                        train_dataset_aug = np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(part) + '.npy')
                        test_dataset = np.load('../../data/interim/AVP/Dataset_Test_0' + str(part) + '.npy')
                    else:
                        train_dataset = np.load('../../data/interim/AVP/Dataset_Train_' + str(part) + '.npy')
                        train_dataset_aug = np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(part) + '.npy')
                        test_dataset = np.load('../../data/interim/AVP/Dataset_Test_' + str(part) + '.npy')

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
                    elif mode=='vae':
                        train_features, _ = model.encode(train_dataset)
                        train_features_aug, _ = model.encode(train_dataset_aug)
                        test_features, _ = model.encode(test_dataset)
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

                    if part<=9:
                        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), train_features)
                        np.save('../../data/processed/' + mode + '/train_features_aug_avp_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), train_features_aug)
                        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), test_features)
                    else:
                        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), train_features)
                        np.save('../../data/processed/' + mode + '/train_features_aug_avp_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), train_features_aug)
                        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), test_features) 

                for part in list_test_participants_lvt:

                    if part<=9:
                        train_dataset = np.load('../../data/interim/LVT/Dataset_Train_0' + str(part) + '.npy')
                        train_dataset_aug = np.load('../../data/interim/LVT/Dataset_Train_Aug_0' + str(part) + '.npy')
                        test_dataset = np.load('../../data/interim/LVT/Dataset_Test_0' + str(part) + '.npy')
                    else:
                        train_dataset = np.load('../../data/interim/LVT/Dataset_Train_' + str(part) + '.npy')
                        train_dataset_aug = np.load('../../data/interim/LVT/Dataset_Train_Aug_' + str(part) + '.npy')
                        test_dataset = np.load('../../data/interim/LVT/Dataset_Test_' + str(part) + '.npy')

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
                    elif mode=='vae':
                        train_features, _ = model.encode(train_dataset)
                        train_features_aug, _ = model.encode(train_dataset_aug)
                        test_features, _ = model.encode(test_dataset)
                    elif mode=='siamese':
                        train_features = model.cnn(train_dataset)
                        train_features_aug = model.cnn(train_dataset_aug)
                        test_features = model.cnn(test_dataset)
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

                    if part<=9:
                        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), train_features)
                        np.save('../../data/processed/' + mode + '/train_features_aug_lvt_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), train_features_aug)
                        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_' + str(latent_dim) + '_0' + str(part) + '_' + str(cv) + '_' + str(it), test_features)
                    else:
                        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), train_features)
                        np.save('../../data/processed/' + mode + '/train_features_aug_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), train_features_aug)
                        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_' + str(latent_dim) + '_' + str(part) + '_' + str(cv) + '_' + str(it), test_features)

                tf.keras.backend.clear_session()
