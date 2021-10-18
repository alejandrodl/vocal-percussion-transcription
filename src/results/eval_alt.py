import os
import numpy as np

import xgboost as xgb
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from warnings import simplefilter
simplefilter(action='ignore')



# Global parameters

list_test_participants_avp = [8,10,18,23]
list_test_participants_lvt = [0,6,7,13]

dataset_names = ['AVP','LVT']

modes = ['eng_mfcc_env','eng_all_classall','eng_all_classred','eng_all_syllall',
         'eng_all_syllred','eng_all_phonall','eng_all_phonred','eng_all_sound',
         'classall','classred','syllall','syllred','phonall','phonred','sound']

num_iterations_algorithms = 5
num_iterations_models = 5
num_cross_validation = 5

percentage_train = 75

# Gradient Boosting Trees parameters

max_depth = 7
min_child_weight = 6
colsample = 0.85
subsample = 0.85
learning_rate = 0.03
reg_lambda = 1.0
reg_alpha = 0
n_estimators = 1000

# KNN parameters

n_neighborss = [3,5,7,9,11]

# Placeholders

accuracies = np.zeros((len(clfs),num_iterations_models,num_cross_validation,num_iterations_algorithms,8))

clfs_array = ['logr','rf','xgboost']

for cl in clfs_array:

    clfs = [cl]

    for mode in modes:

        if not os.path.isdir('results/' + mode):
            os.mkdir('results/' + mode)

        count_part = 0

        print('Calculating results for ' + mode + ' and ' + clfs[0] + ' classifier...')

        for dataset_name in dataset_names:

            if dataset_name=='AVP':
                list_test = list_test_participants_avp
            elif dataset_name=='LVT':
                list_test = list_test_participants_lvt

            for part in range(len(list_test)):

                Part = list_test[part]

                for cv in range(num_cross_validation):

                    for it_mod in range(num_iterations_models):

                        # Load and process spectrograms

                        print('\n')
                        print('Hyperparameters: ' + str([Part,it_mod]))
                        print('\n')

                        if dataset_name=='AVP':

                            if Part<=9:
                                classes_str = np.load('data/interim/AVP/Classes_Test_0' + str(Part) + '.npy')
                            else:
                                classes_str = np.load('data/interim/AVP/Classes_Test_' + str(Part) + '.npy')

                            classes_eval = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='kd':
                                    classes_eval[n] = 0
                                elif classes_str[n]=='sd':
                                    classes_eval[n] = 1
                                elif classes_str[n]=='hhc':
                                    classes_eval[n] = 2
                                elif classes_str[n]=='hho':
                                    classes_eval[n] = 3

                            if Part<=9:
                                classes_str = np.load('data/interim/AVP/Classes_Train_Aug_0' + str(Part) + '.npy')
                            else:
                                classes_str = np.load('data/interim/AVP/Classes_Train_Aug_' + str(Part) + '.npy')

                            classes = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='kd':
                                    classes[n] = 0
                                elif classes_str[n]=='sd':
                                    classes[n] = 1
                                elif classes_str[n]=='hhc':
                                    classes[n] = 2
                                elif classes_str[n]=='hho':
                                    classes[n] = 3

                            if 'eng_all' not in mode:

                                if mode!='eng_mfcc_env':

                                    if Part<=9:
                                        dataset = np.load('data/processed/' + mode + '/train_features_aug_avp_' + mode + '_32_0' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_32_0' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    else:
                                        dataset = np.load('data/processed/' + mode + '/train_features_aug_avp_' + mode + '_32_' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_32_' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                            
                                else:

                                    if Part<=9:
                                        dataset = np.load('data/processed/' + mode + '/train_features_avp_' + mode + '_32_0' + str(Part) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_32_0' + str(Part) + '.npy')
                                    else:
                                        dataset = np.load('data/processed/' + mode + '/train_features_avp_' + mode + '_32_' + str(Part) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_32_' + str(Part) + '.npy')

                            else:

                                if Part<=9:
                                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_avp_' + mode[:7] + '_0' + str(Part) + '.npy')
                                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_avp_' + mode[:7] + '_0' + str(Part) + '.npy')
                                else:
                                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_avp_' + mode[:7] + '_' + str(Part) + '.npy')
                                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_avp_' + mode[:7] + '_' + str(Part) + '.npy')

                                if 'phon' not in mode:

                                    indices_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')[:32]
                                    indices_selected = indices_selected.tolist()
                                    
                                    dataset = dataset[:,indices_selected]
                                    dataset_eval = dataset_eval[:,indices_selected]

                                else:

                                    indices_onset_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_onset_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    indices_onset_selected = indices_onset_selected.tolist()

                                    indices_nucleus_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_nucleus_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    indices_nucleus_selected = indices_nucleus_selected.tolist()

                                    cutoff = 32
                                    indices_selected = []
                                    while len(indices_selected)<=32:
                                        indices_selected = list(set(indices_onset_selected[:cutoff])&set(indices_nucleus_selected[:cutoff]))
                                        cutoff += 1
                                    indices_selected = indices_selected[:32]
                                    
                                    dataset = dataset[:,indices_selected]
                                    dataset_eval = dataset_eval[:,indices_selected]

                        elif dataset_name=='LVT':

                            if Part<=9:
                                classes_str = np.load('data/interim/LVT/Classes_Test_0' + str(Part) + '.npy')
                            else:
                                classes_str = np.load('data/interim/LVT/Classes_Test_' + str(Part) + '.npy')

                            classes_eval = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='Kick':
                                    classes_eval[n] = 0
                                elif classes_str[n]=='Snare':
                                    classes_eval[n] = 1
                                elif classes_str[n]=='HH':
                                    classes_eval[n] = 2

                            if Part<=9:
                                classes_str = np.load('data/interim/LVT/Classes_Train_Aug_0' + str(Part) + '.npy')
                            else:
                                classes_str = np.load('data/interim/LVT/Classes_Train_Aug_' + str(Part) + '.npy')

                            classes = np.zeros(len(classes_str))
                            for n in range(len(classes_str)):
                                if classes_str[n]=='Kick':
                                    classes[n] = 0
                                elif classes_str[n]=='Snare':
                                    classes[n] = 1
                                elif classes_str[n]=='HH':
                                    classes[n] = 2

                            if 'eng_all' not in mode:

                                if mode!='eng_mfcc_env':

                                    if Part<=9:
                                        dataset = np.load('data/processed/' + mode + '/train_features_aug_lvt_' + mode + '_16_0' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_lvt_' + mode + '_16_0' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    else:
                                        dataset = np.load('data/processed/' + mode + '/train_features_aug_lvt_' + mode + '_16_' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_lvt_' + mode + '_16_' + str(Part) + '_' + str(cv) + '_' + str(it_mod) + '.npy')

                                else:

                                    if Part<=9:
                                        dataset = np.load('data/processed/' + mode + '/train_features_lvt_' + mode + '_16_0' + str(Part) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_lvt_' + mode + '_16_0' + str(Part) + '.npy')
                                    else:
                                        dataset = np.load('data/processed/' + mode + '/train_features_lvt_' + mode + '_16_' + str(Part) + '.npy')
                                        dataset_eval = np.load('data/processed/' + mode + '/test_features_lvt_' + mode + '_16_' + str(Part) + '.npy')

                            else:

                                if Part<=9:
                                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_lvt_' + mode[:7] + '_0' + str(Part) + '.npy')
                                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_lvt_' + mode[:7] + '_0' + str(Part) + '.npy')
                                else:
                                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_lvt_' + mode[:7] + '_' + str(Part) + '.npy')
                                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_lvt_' + mode[:7] + '_' + str(Part) + '.npy')

                                if 'phon' not in mode:

                                    indices_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')[:16]
                                    indices_selected = indices_selected.tolist()
                                    
                                    dataset = dataset[:,indices_selected]
                                    dataset_eval = dataset_eval[:,indices_selected]

                                else:

                                    indices_onset_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_onset_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    indices_onset_selected = indices_onset_selected.tolist()

                                    indices_nucleus_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_nucleus_eng_' + mode[8:] + '_' + str(cv) + '_' + str(it_mod) + '.npy')
                                    indices_nucleus_selected = indices_nucleus_selected.tolist()

                                    cutoff = 16
                                    indices_selected = []
                                    while len(indices_selected)<=16:
                                        indices_selected = list(set(indices_onset_selected[:cutoff])&set(indices_nucleus_selected[:cutoff]))
                                        cutoff += 1
                                    indices_selected = indices_selected[:16]
                                    
                                    dataset = dataset[:,indices_selected]
                                    dataset_eval = dataset_eval[:,indices_selected]
                        
                        for feat in range(dataset.shape[-1]):
                            mean = np.mean(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                            std = np.std(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                            dataset[:,feat] = (dataset[:,feat]-mean)/(std+1e-16)
                            dataset_eval[:,feat] = (dataset_eval[:,feat]-mean)/(std+1e-16)

                        mean = np.mean(np.vstack((dataset,dataset_eval)))
                        std = np.std(np.vstack((dataset,dataset_eval)))
                        dataset = (dataset-mean)/(std+1e-16)
                        dataset_eval = (dataset_eval-mean)/(std+1e-16)

                        np.random.seed(0)
                        np.random.shuffle(dataset)

                        np.random.seed(0)
                        np.random.shuffle(classes)

                        cutoff_train = int((percentage_train/100)*dataset.shape[0])

                        dataset = dataset.astype('float32')
                        classes = classes.astype('float32')
                        dataset_eval = dataset_eval.astype('float32')
                        classes_eval = classes_eval.astype('float32')

                    for b in range(len(clfs)):

                        clf = clfs[b]

                        if clf=='logr':

                            for it_alg in range(num_iterations_algorithms):

                                clf = LogisticRegression(solver='liblinear')
                                clf.fit(dataset, classes)

                                accuracies[b,it_mod,cv,it_alg,count_part] = clf.score(dataset_eval, classes_eval)

                                #print(accuracies[b,it_mod,cv,it_alg,count_part])

                        elif clf=='rf':

                            for it_alg in range(num_iterations_algorithms):

                                clf = RandomForestClassifier(random_state=it_alg)
                                clf.fit(dataset, classes)

                                accuracies[b,it_mod,cv,it_alg,count_part] = clf.score(dataset_eval, classes_eval)
                                
                                #print(accuracies[b,it_mod,cv,it_alg,count_part])

                        elif clf=='xgboost':

                            for it_alg in range(num_iterations_algorithms):

                                params = {'max_depth':max_depth,
                                            'min_child_weight': min_child_weight,
                                            'learning_rate':learning_rate,
                                            'subsample': subsample,
                                            'colsample_bytree': colsample,
                                            'objective':'multi:softmax',
                                            'reg_lambda':reg_lambda,
                                            'reg_alpha':reg_alpha,
                                            'n_estimators':n_estimators,
                                            'random_state':it_alg}

                                model = xgb.XGBClassifier(**params)
                                model.fit(dataset, classes, eval_metric='merror')

                                accuracies[b,it_mod,cv,it_alg,count_part] = accuracy_score(classes_eval, model.predict(dataset_eval))

                                #print(accuracies[b,it_mod,cv,it_alg,count_part])

                count_part += 1

        np.save('results/' + mode + '/accuracies', accuracies)

    # Calculate boxeme-wise weights

    num_test_boxemes = []
    for part in list_test_participants_avp:
        if part<=9:
            test_dataset = np.load('data/interim/AVP/Dataset_Test_0' + str(part) + '.npy')
        else:
            test_dataset = np.load('data/interim/AVP/Dataset_Test_' + str(part) + '.npy')
        num_test_boxemes.append(test_dataset.shape[0])
    for part in list_test_participants_lvt:
        if part<=9:
            test_dataset = np.load('data/interim/LVT/Dataset_Test_0' + str(part) + '.npy')
        else:
            test_dataset = np.load('data/interim/LVT/Dataset_Test_' + str(part) + '.npy')
        num_test_boxemes.append(test_dataset.shape[0])
    boxeme_wise_weights = num_test_boxemes/np.sum(np.array(num_test_boxemes))

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
                            accuracies_raw[i,j,k,l] *= boxeme_wise_weights*8

            accuracies_mean = np.mean(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
            accuracies_std = np.std(np.mean(np.mean(np.mean(accuracies_raw,axis=-1),axis=-1),axis=-1),axis=-1)
            
            print([mode,clf])
            print([accuracies_mean[0],accuracies_std[0]])
                            