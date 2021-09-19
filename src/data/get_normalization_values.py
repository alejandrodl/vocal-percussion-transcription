import os
import numpy as np



pretrain_dataset = np.zeros((1, 64, 48))
for n in range(28):
    if n<=9:
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '.npy')))
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_Aug_0' + str(n) + '.npy')))
    else:
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Train_Aug_' + str(n) + '.npy')))
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/AVP/Dataset_Test_Aug_' + str(n) + '.npy')))
for n in range(20):
    if n<=9:
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_0' + str(n) + '.npy')))
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_Aug_0' + str(n) + '.npy')))
    else:
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Train_Aug_' + str(n) + '.npy')))
        pretrain_dataset = np.vstack((pretrain_dataset, np.load('data/interim/LVT/Dataset_Test_Aug_' + str(n) + '.npy')))
pretrain_dataset = pretrain_dataset[1:]

norm_min_1 = np.min(pretrain_dataset)
norm_max_1 = np.max(pretrain_dataset)

pretrain_dataset = (pretrain_dataset-np.min(pretrain_dataset))/(np.max(pretrain_dataset)-np.min(pretrain_dataset)+1e-16)
pretrain_dataset = np.log(pretrain_dataset+1e-4)

norm_min_2 = np.min(pretrain_dataset)
norm_max_2 = np.max(pretrain_dataset)

print(norm_min_1)
print(norm_max_1)

print(norm_min_2)
print(norm_max_2)

