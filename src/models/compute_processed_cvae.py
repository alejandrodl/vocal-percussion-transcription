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



os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_logical_devices('GPU')
#print(gpus)

percentage_train = 90

epochs = 10000
patience_lr = 5
patience_early = 10

batch_size = 1024
num_examples_to_generate = 16

#warnings.simplefilter(action='ignore', category=FutureWarning)

dropout = 0.5
num_filters = 64
num_filters_str = '64'
latent_dim = 16

frame_sizes = ['1024']
lrs = [1e-3]
kernel_heights = [3, 5, 7]
kernel_widths = [3, 5, 7]

best_params_and_losses = np.zeros((len(lrs),len(kernel_heights),len(kernel_widths),5,4))
    
# Main

frame_size = '1024'
lr = 1e-3
kernel_height = 3
kernel_width = 3

Pretrain_Dataset = np.zeros((1, 32, num_filters))

# AVP Personal

for n in range(28):
    if n<=9:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_AVP/Dataset_Train_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_AVP/Dataset_Test_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
    else:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_AVP/Dataset_Train_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_AVP/Dataset_Test_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))

# AVP Fixed Small

Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_AVPFS/Dataset_AVP_Fixed_Small_Aug_' + num_filters_str + '_' + frame_size + '.npy')))

# LVT 1

for n in range(20):
    if n<=9:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_1_Train_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_1_Test_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
    else:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_1_Train_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_1_Test_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))

# LVT 2

for n in range(20):
    if n<=9:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_2_Train_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_2_Test_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
    else:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_2_Train_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_2_Test_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))

# LVT 3

for n in range(20):
    if n<=9:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_3_Train_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_3_Test_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
    else:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_3_Train_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_LVT/Dataset_3_Test_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))

# BTX

for n in range(14):
    if n<=9:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_BTX/Dataset_BTX_Aug_0' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))
    else:
        Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_BTX/Dataset_BTX_Aug_' + str(n) + '_' + num_filters_str + '_' + frame_size + '.npy')))

# VIM

Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_VIM/Dataset_VIM_Percussive_Aug_' + num_filters_str + '_' + frame_size + '.npy')))

# FSB Multi

Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_FSB/Dataset_FSB_Multi_Aug_' + num_filters_str + '_' + frame_size + '.npy')))

# FSB Single

Pretrain_Dataset = np.vstack((Pretrain_Dataset, np.load('../Data/UC_FSB/Dataset_FSB_Single_Aug_' + num_filters_str + '_' + frame_size + '.npy')))

# Remove first data point and fit to batch size

Pretrain_Dataset = Pretrain_Dataset[1:]
print(Pretrain_Dataset.shape)

L = Pretrain_Dataset.shape[0]
train_ratio = percentage_train/100

Pretrain_Dataset = Pretrain_Dataset[:(Pretrain_Dataset.shape[0]-(Pretrain_Dataset.shape[0]%batch_size))]
print(Pretrain_Dataset.shape)

cutoff_train = int(int(train_ratio*int(L/batch_size))*batch_size)

#Pretrain_Dataset = np.zeros((512,32,64))

Classes_Train = np.zeros(Pretrain_Dataset.shape[0])

Pretrain_Dataset = (Pretrain_Dataset-np.min(Pretrain_Dataset))/(np.max(Pretrain_Dataset)-np.min(Pretrain_Dataset))
print([np.min(Pretrain_Dataset),np.max(Pretrain_Dataset)])
Pretrain_Dataset = np.log(Pretrain_Dataset+1e-4)
Pretrain_Dataset = (Pretrain_Dataset-np.min(Pretrain_Dataset))/(np.max(Pretrain_Dataset)-np.min(Pretrain_Dataset))
print([np.mean(Pretrain_Dataset),np.std(Pretrain_Dataset)])

train_images = Pretrain_Dataset[:cutoff_train]
test_images = Pretrain_Dataset[cutoff_train:]

print(train_images.shape)
print(test_images.shape)

train_images = np.expand_dims(train_images,axis=-1).astype('float32')
test_images = np.expand_dims(test_images,axis=-1).astype('float32')

train_size = train_images.shape[0]
test_size = test_images.shape[0]

fix_seeds(0)

np.random.seed(0)
np.random.shuffle(train_images)

np.random.seed(0)
np.random.shuffle(test_images)

for it in range(5):

    model = CVAE(latent_dim, batch_size, kernel_height, kernel_width, dropout)
    #model = CVAE(latent_dim, kernel_height, kernel_width, dropout)

    optimizer = tf.keras.optimizers.Adam(lr)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.000001, cooldown=0, min_lr=0)

    with tf.device(gpu_name):

        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=False)
        history = model.fit(train_images, train_images, batch_size=batch_size, epochs=epochs, validation_data=(test_images,test_images), callbacks=[early_stopping,lr_scheduler], shuffle=True)  # , verbose=0

        fpath = 'best_models/final_model_' + frame_size
        model.save_weights(fpath+'.h5')


