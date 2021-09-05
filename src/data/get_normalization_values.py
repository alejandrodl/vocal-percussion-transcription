import os
import numpy as np



frame_sizes = ['512','1024','2048']

norm_min_max_1 = np.zeros((len(frame_sizes),2))
norm_min_max_2 = np.zeros((len(frame_sizes),2))

for a in range(len(frame_sizes)):

    frame_size = frame_sizes[a]

    pretrain_dataset = np.zeros((1, 64, 64))
    for n in range(28):
        print(n)
        if n<=9:
            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_0' + str(n) + '_' + frame_size + '.npy')))
            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_0' + str(n) + '_' + frame_size + '.npy')))
        else:
            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Train_Aug_' + str(n) + '_' + frame_size + '.npy')))
            pretrain_dataset = np.vstack((pretrain_dataset, np.load('../../data/interim/AVP/Dataset_Test_Aug_' + str(n) + '_' + frame_size + '.npy')))
    pretrain_dataset = pretrain_dataset[1:]

    norm_min_max_1[a][0] = np.min(pretrain_dataset)
    norm_min_max_1[a][1] = np.max(pretrain_dataset)

    pretrain_dataset = (pretrain_dataset-np.min(pretrain_dataset))/(np.max(pretrain_dataset)-np.min(pretrain_dataset)+1e-16)
    pretrain_dataset = np.log(pretrain_dataset+1e-4)

    norm_min_max_2[a][0] = np.min(pretrain_dataset)
    norm_min_max_2[a][1] = np.max(pretrain_dataset)

print(norm_min_max_1)
print(norm_min_max_2)

np.save('../../data/offline_norm_min_max_1', norm_min_max_1)
np.save('../../data/offline_norm_min_max_2', norm_min_max_2)