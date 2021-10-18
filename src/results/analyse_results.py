import numpy as np


# Evaluate

modes = ['eng_mfcc_env','eng_all_classall','eng_all_classred','eng_all_syllall',
         'eng_all_syllred','eng_all_phonall','eng_all_phonred','eng_all_sound',
         'classall','classred','syllall','syllred','phonall','phonred','sound']
clfs = ['knn']

list_test_participants_avp = [8,10,18,23]
list_test_participants_lvt = [0,6,7,13]

# Calculate utterance-wise weights

num_test_utterances = []
for part in list_test_participants_avp:
    if part<=9:
        test_dataset = np.load('data/interim/AVP/Dataset_Test_0' + str(part) + '.npy')
    else:
        test_dataset = np.load('data/interim/AVP/Dataset_Test_' + str(part) + '.npy')
    num_test_utterances.append(test_dataset.shape[0])
for part in list_test_participants_lvt:
    if part<=9:
        test_dataset = np.load('data/interim/LVT/Dataset_Test_0' + str(part) + '.npy')
    else:
        test_dataset = np.load('data/interim/LVT/Dataset_Test_' + str(part) + '.npy')
    num_test_utterances.append(test_dataset.shape[0])
utterance_wise_weights = num_test_utterances/np.sum(np.array(num_test_utterances))

# Results participant-wise

print('\n')
print('Participant-wise')
print('\n')

for b in range(len(modes)):

    for c in range(len(clfs)):

        mode = modes[b]
        clf = clfs[c]

        accuracies_raw = np.load('results/' + mode + '/accuracies.npy')
        accuracies_mean = np.mean(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
        accuracies_std = np.std(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
        
        print([mode,clf])
        print([accuracies_mean[0],accuracies_std[0]])

# Results boxeme-wise

print('\n')
print('Boxeme-wise')
print('\n')

for b in range(len(modes)):

    for c in range(len(clfs)):

        mode = modes[b]
        clf = clfs[c]

        accuracies_raw = np.load('results/' + mode + '/accuracies.npy')

        for i in range(accuracies_raw.shape[0]):
            for j in range(accuracies_raw.shape[1]):
                for k in range(accuracies_raw.shape[2]):
                    for l in range(accuracies_raw.shape[3]):
                        accuracies_raw[i,j,k,l] *= utterance_wise_weights*8

        accuracies_mean = np.mean(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
        accuracies_std = np.std(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
        
        print([mode,clf])
        print([accuracies_mean[0],accuracies_std[0]])