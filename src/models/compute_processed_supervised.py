import os
import PIL
import glob
import time
import torch
import random
import numpy as np
import scipy as sp

from torch import nn
import torch.nn as nn
import torch.utils.data
from torch.utils import data
import torch.nn.functional as F
import torch.utils.data as utils
from IPython.display import clear_output
from torchvision.utils import save_image

import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)
import tensorflow_probability as tfp

import optuna

from utils import *
from networks_tf import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.nice(0)
gpu_name = '/GPU:0'
#gpu_name = '/device:CPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters

mode = 'classall'

percentage_train = 80

epochs = 10000
patience_lr = 10
patience_early = 20

batch_size = 128
num_examples_to_generate = 16

num_filters = 64
num_filters_str = '64'
latent_dim = 16

frame_sizes = ['1024', '2048', '4096']
kernel_heights = [3, 5, 7]
kernel_widths = [3, 5, 7]

lr = 1e-3

part_starts = [0,4,8,12,16,20,24]
part_ends = [4,8,12,16,20,24,28]

for a in range(len(frame_sizes)):
    
    frame_size = frame_sizes[a]
        
    for b in range(len(kernel_heights)):

        for c in range(len(kernel_widths)):
                
            kernel_height = kernel_heights[b]
            kernel_width = kernel_widths[c]
            
            for part in range(len(part_starts)):

                part_start = part_starts[part]
                part_end = part_ends[part]

                print('/n')
                print('Hyperparameters: ' + str([frame_size,kernel_height,kernel_width,part]))
                print('/n')

                Classes_Str = np.zeros(1)
                for n in range(28):
                    if n>=part_start and n<part_end:
                        continue
                    else:
                        if n<=9:
                            Classes_Str = np.concatenate((Classes_Str, np.load('../Data/UC_AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                            Classes_Str = np.concatenate((Classes_Str, np.load('../Data/UC_AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                        else:
                            Classes_Str = np.concatenate((Classes_Str, np.load('../Data/UC_AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                            Classes_Str = np.concatenate((Classes_Str, np.load('../Data/UC_AVP/Classes_Test_Aug_' + str(n) + '.npy')))
                Classes_Str = Classes_Str[1:]

                Classes = np.zeros(len(Classes_Str))
                for n in range(len(Classes_Str)):
                    if Classes_Str[n]=='kd':
                        Classes[n] = 0
                    elif Classes_Str[n]=='sd':
                        Classes[n] = 1
                    elif Classes_Str[n]=='hhc':
                        Classes[n] = 2
                    elif Classes_Str[n]=='hho':
                        Classes[n] = 3
                        
                Pretrain_Dataset = np.zeros((1,256))
                for n in range(28):
                    if n>=part_start and n<part_end:
                        continue
                    else:
                        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('features/Features_Train_Aug_' + frame_size + '_' + str(kernel_height) + '_' + str(kernel_width) + '_' + str(n) + '.npy')))
                        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('features/Features_Test_Aug_' + frame_size + '_' + str(kernel_height) + '_' + str(kernel_width) + '_' + str(n) + '.npy')))
                Pretrain_Dataset = Pretrain_Dataset[1:]
                
                print(Pretrain_Dataset.shape)
                print(Classes.shape)

                # Fit to batch size

                L = Pretrain_Dataset.shape[0]
                train_ratio = percentage_train/100

                Pretrain_Dataset = Pretrain_Dataset[:(Pretrain_Dataset.shape[0]-(Pretrain_Dataset.shape[0]%batch_size))]
                Classes = Classes[:(Classes.shape[0]-(Classes.shape[0]%batch_size))]
                
                np.random.seed(0)
                np.random.shuffle(Pretrain_Dataset)

                np.random.seed(0)
                np.random.shuffle(Classes)

                cutoff_train = int(int(train_ratio*int(L/batch_size))*batch_size)
                
                train_classes = Classes[:cutoff_train].astype('float32')
                test_classes = Classes[cutoff_train:].astype('float32')

                train_features = Pretrain_Dataset[:cutoff_train]
                test_features = Pretrain_Dataset[cutoff_train:]

                train_features = np.expand_dims(train_features,axis=-1).astype('float32')
                test_features = np.expand_dims(test_features,axis=-1).astype('float32')

                train_size = train_features.shape[0]
                test_size = test_features.shape[0]
                
                with tf.device(gpu_name):

                    model = MLP_Class(np.max(Classes)+1, 256)

                    optimizer = tf.keras.optimizers.Adam(lr)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience_early, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=patience_lr, verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

                    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                    history = model.fit(train_features, train_classes, batch_size=batch_size, epochs=epochs, validation_data=(test_features,test_classes), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0

                    fpath = 'best_models/model_fine_' + mode + '_' + frame_size + '_' + str(kernel_height) + '_' + str(kernel_width) + '_' + str(part)
                    model.save_weights(fpath+'.h5')