import os
import sys
import numpy as np
import tensorflow as tf
from itertools import combinations
import tensorflow_probability as tfp

#sys.path.append('/homes/adl30/vocal-percussion-transcription')
#from src.utils import *

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Global parameters

num_crossval = 5
num_iterations = 5

percentage_train = 80
modes = ['classall','classred','syllall','syllred','phonall','phonred','sound']
#modes = ['phonall','phonred','sound']
mode_feat = 'eng_all'

list_test_participants_avp = [8,10,18,23]
list_test_participants_lvt = [0,6,7,13]

# Data parameters

frame_size = '1024'

# Features

features_env = ['Env_DerAvAfterMax','Env_MaxDerBeforeMax','Env_Flatness','Env_TCToTotal']
features_mfcc = ['MFCC_0_mean','MFCC_0_std','MFCC_0_dmean','MFCC_0_dvar','MFCC_0_ddmean','MFCC_0_ddvar',
                'MFCC_1_mean','MFCC_1_std','MFCC_1_dmean','MFCC_1_dvar','MFCC_1_ddmean','MFCC_1_ddvar',
                'MFCC_2_mean','MFCC_2_std','MFCC_2_dmean','MFCC_2_dvar','MFCC_2_ddmean','MFCC_2_ddvar',
                'MFCC_3_mean','MFCC_3_std','MFCC_3_dmean','MFCC_3_dvar','MFCC_3_ddmean','MFCC_3_ddvar',
                'MFCC_4_mean','MFCC_4_std','MFCC_4_dmean','MFCC_4_dvar','MFCC_4_ddmean','MFCC_4_ddvar',
                'MFCC_5_mean','MFCC_5_std','MFCC_5_dmean','MFCC_5_dvar','MFCC_5_ddmean','MFCC_5_ddvar',
                'MFCC_6_mean','MFCC_6_std','MFCC_6_dmean','MFCC_6_dvar','MFCC_6_ddmean','MFCC_6_ddvar',
                'MFCC_7_mean','MFCC_7_std','MFCC_7_dmean','MFCC_7_dvar','MFCC_7_ddmean','MFCC_7_ddvar',
                'MFCC_8_mean','MFCC_8_std','MFCC_8_dmean','MFCC_8_dvar','MFCC_8_ddmean','MFCC_8_ddvar',
                'MFCC_9_mean','MFCC_9_std','MFCC_9_dmean','MFCC_9_dvar','MFCC_9_ddmean','MFCC_9_ddvar',
                'MFCC_10_mean','MFCC_10_std','MFCC_10_dmean','MFCC_10_dvar','MFCC_10_ddmean','MFCC_10_ddvar',
                'MFCC_11_mean','MFCC_11_std','MFCC_11_dmean','MFCC_11_dvar','MFCC_11_ddmean','MFCC_11_ddvar',
                'MFCC_12_mean','MFCC_12_std','MFCC_12_dmean','MFCC_12_dvar','MFCC_12_ddmean','MFCC_12_ddvar']
features_melbands = ['MelBand_0','MelBand_1','MelBand_2','MelBand_3','MelBand_4','MelBand_5','MelBand_6','MelBand_7',
                    'MelBand_8','MelBand_9','MelBand_10','MelBand_11','MelBand_12','MelBand_13','MelBand_14','MelBand_15',
                    'MelBand_16','MelBand_17','MelBand_18','MelBand_19','MelBand_20','MelBand_21','MelBand_22','MelBand_23',
                    'MelBand_24','MelBand_25','MelBand_26','MelBand_27','MelBand_28','MelBand_29','MelBand_30','MelBand_31',
                    'MelBand_32','MelBand_33','MelBand_34','MelBand_35','MelBand_36','MelBand_37','MelBand_38','MelBand_39']
features_misc = ['RollOff_25_mean','RollOff_50_mean','RollOff_90_mean','RollOff_95_mean','SpecComplexity_mean','HFC_mean','StrongPeak_mean','SpecCentroid_mean','SpecVariance_mean','SpecSkewness_mean','SpecKurtosis_mean','SpecCrest_mean','SpecDecrease_mean','SpecEntropy_mean','SpecFlatness_mean','SpecRMS_mean','ZCR_mean',
                    'RollOff_25_var','RollOff_50_var','RollOff_90_var','RollOff_95_var','SpecComplexity_var','HFC_var','StrongPeak_var','SpecCentroid_var','SpecVariance_var','SpecSkewness_var','SpecKurtosis_var','SpecCrest_var','SpecDecrease_var','SpecEntropy_var','SpecFlatness_var','SpecRMS_var','ZCR_var',
                    'RollOff_25_min','RollOff_50_min','RollOff_90_min','RollOff_95_min','SpecComplexity_min','HFC_min','StrongPeak_min','SpecCentroid_min','SpecVariance_min','SpecSkewness_min','SpecKurtosis_min','SpecCrest_min','SpecDecrease_min','SpecEntropy_min','SpecFlatness_min','SpecRMS_min','ZCR_min',
                    'RollOff_25_max','RollOff_50_max','RollOff_90_max','RollOff_95_max','SpecComplexity_max','HFC_max','StrongPeak_max','SpecCentroid_max','SpecVariance_max','SpecSkewness_max','SpecKurtosis_max','SpecCrest_max','SpecDecrease_max','SpecEntropy_max','SpecFlatness_max','SpecRMS_max','ZCR_max',
                    'RollOff_25_dmean','RollOff_50_dmean','RollOff_90_dmean','RollOff_95_dmean','SpecComplexity_dmean','HFC_dmean','StrongPeak_dmean','SpecCentroid_dmean','SpecVariance_dmean','SpecSkewness_dmean','SpecKurtosis_dmean','SpecCrest_mean','SpecDecrease_dmean','SpecEntropy_dmean','SpecFlatness_dmean','SpecRMS_dmean','ZCR_dmean',
                    'RollOff_25_dvar','RollOff_50_dvar','RollOff_90_dvar','RollOff_95_dvar','SpecComplexity_dvar','HFC_dvar','StrongPeak_dvar','SpecCentroid_dvar','SpecVariance_dvar','SpecSkewness_dvar','SpecKurtosis_dvar','SpecCrest_dvar','SpecDecrease_dvar','SpecEntropy_dvar','SpecFlatness_dvar','SpecRMS_dvar','ZCR_dvar',
                    'RollOff_25_dmin','RollOff_50_dmin','RollOff_90_dmin','RollOff_95_dmin','SpecComplexity_dmin','HFC_dmin','StrongPeak_dmin','SpecCentroid_dmin','SpecVariance_dmin','SpecSkewness_dmin','SpecKurtosis_dmin','SpecCrest_dmin','SpecDecrease_dmin','SpecEntropy_dmin','SpecFlatness_dmin','SpecRMS_dmin','ZCR_dmin',
                    'RollOff_25_dmax','RollOff_50_dmax','RollOff_90_dmax','RollOff_95_dmax','SpecComplexity_dmax','HFC_dmax','StrongPeak_dmax','SpecCentroid_dmax','SpecVariance_dmax','SpecSkewness_dmax','SpecKurtosis_dmax','SpecCrest_dmax','SpecDecrease_dmax','SpecEntropy_dmax','SpecFlatness_dmax','SpecRMS_dmax','ZCR_dmax']

features_names = features_env + features_mfcc + features_melbands + features_misc
features_names = np.array(features_names)

# Main loop

for m in range(len(modes)):

    mode = modes[m]

    if not os.path.isdir('data/processed/' + mode):
        os.mkdir('data/processed/' + mode)

    print('\n')
    print(mode)
    print('\n')

    # Load and process spectrograms

    dataset = np.zeros((1,258))
    for part in range(28):
        if part in list_test_participants_avp:
            continue
        else:
            if part<=9:
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/train_features_avp_' + mode_feat + '_0' + str(part) + '.npy')))
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/test_features_avp_' + mode_feat + '_0' + str(part) + '.npy')))
            else:
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/train_features_avp_' + mode_feat + '_' + str(part) + '.npy')))
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/test_features_avp_' + mode_feat + '_' + str(part) + '.npy')))
    for part in range(20):
        if part in list_test_participants_lvt:
            continue
        else:
            if part<=9:
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/train_features_lvt_' + mode_feat + '_0' + str(part) + '.npy')))
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/test_features_lvt_' + mode_feat + '_0' + str(part) + '.npy')))
            else:
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/train_features_lvt_' + mode_feat + '_' + str(part) + '.npy')))
                dataset = np.vstack((dataset, np.load('data/processed/' + mode_feat + '/test_features_lvt_' + mode_feat + '_' + str(part) + '.npy')))
    dataset = dataset[1:]

    for feat in range(dataset.shape[-1]):
        mean = np.mean(dataset[:,feat])
        std = np.std(dataset[:,feat])
        dataset[:,feat] = (dataset[:,feat]-mean)/(std+1e-16)

    cutoff_test = int(((100-percentage_train)/100)*dataset.shape[0])
    
    np.random.seed(0)
    np.random.shuffle(dataset)

    # Load and process classes

    if 'syllall' in mode or 'phonall' in mode:

        classes_onset = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_' + str(n) + '.npy')))
        classes_onset = classes_onset[1:]

        classes_nucleus = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_' + str(n) + '.npy')))
        classes_nucleus = classes_nucleus[1:]

        num_onset = np.max(classes_onset)+1
        num_nucleus = np.max(classes_nucleus)+1

    elif 'syllred' in mode or 'phonred' in mode:

        classes_onset = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_' + str(n) + '.npy')))
        classes_onset = classes_onset[1:]

        classes_nucleus = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_' + str(n) + '.npy')))
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
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_' + str(n) + '.npy')))
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
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_' + str(n) + '.npy')))
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

    elif 'sound' in mode:

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

        classes_pre_siamese_onset = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_pre_siamese_onset = np.concatenate((classes_pre_siamese_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                else:
                    classes_pre_siamese_onset = np.concatenate((classes_pre_siamese_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_pre_siamese_onset = np.concatenate((classes_pre_siamese_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                else:
                    classes_pre_siamese_onset = np.concatenate((classes_pre_siamese_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
        classes_pre_siamese_onset = classes_pre_siamese_onset[1:]

        classes_pre_siamese_nucleus = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_pre_siamese_nucleus = np.concatenate((classes_pre_siamese_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                else:
                    classes_pre_siamese_nucleus = np.concatenate((classes_pre_siamese_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_pre_siamese_nucleus = np.concatenate((classes_pre_siamese_nucleus, np.load('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                else:
                    classes_pre_siamese_nucleus = np.concatenate((classes_pre_siamese_nucleus, np.load('data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
        classes_pre_siamese_nucleus = classes_pre_siamese_nucleus[1:]

        cmb = []
        classes_syllables = np.zeros(len(classes_pre_siamese_onset))
        for n in range(len(classes_pre_siamese_onset)):
            combination = [classes_pre_siamese_onset[n],classes_pre_siamese_nucleus[n]]
            if combination not in cmb:
                cmb.append(combination)
                classes_syllables[n] = cmb.index(combination)
            else:
                classes_syllables[n] = cmb.index(combination)

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

        np.random.seed(0)
        np.random.shuffle(classes)

        for cv in range(num_crossval):

            for it in range(num_iterations):

                dataset_cv = dataset.copy()
                classes_cv = classes.copy()

                idx_start = cv*cutoff_test
                idx_end = (cv+1)*cutoff_test

                idxs_test = np.arange(idx_start,idx_end).tolist()

                dataset_train = np.delete(dataset_cv,idxs_test,axis=0).astype('float32')
                dataset_test = dataset_cv[idx_start:idx_end].astype('float32')

                classes_train = np.delete(classes_cv,idxs_test,axis=0).astype('float32')
                classes_test = classes_cv[idx_start:idx_end].astype('float32')

                print('Calculating importances...')

                forest = RandomForestClassifier(n_estimators=100,random_state=it,n_jobs=-1)

                forest.fit(dataset_train, classes_train)
                results = permutation_importance(forest, dataset_test, classes_test, n_repeats=5, random_state=it)

                indices_sorted = np.array(results.importances_mean).argsort()[::-1]
                importances_sorted = np.sort(np.array(results.importances_mean))[::-1]
                names_sorted = features_names[indices_sorted.tolist()]

                np.save('data/processed/' + mode + '/names_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), names_sorted)
                np.save('data/processed/' + mode + '/indices_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), indices_sorted)
                np.save('data/processed/' + mode + '/importances_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), importances_sorted)

    elif 'phon' in mode:

        np.random.seed(0)
        np.random.shuffle(classes_onset)

        np.random.seed(0)
        np.random.shuffle(classes_nucleus)

        for cv in range(num_crossval):

            for it in range(num_iterations):

                dataset_cv = dataset.copy()
                classes_onset_cv = classes_onset.copy()
                classes_nucleus_cv = classes_nucleus.copy()

                idx_start = cv*cutoff_test
                idx_end = (cv+1)*cutoff_test

                idxs_test = np.arange(idx_start,idx_end).tolist()

                dataset_train = np.delete(dataset_cv,idxs_test,axis=0).astype('float32')
                dataset_test = dataset_cv[idx_start:idx_end].astype('float32')

                classes_train_onset = np.delete(classes_onset_cv,idxs_test,axis=0).astype('float32')
                classes_train_nucleus = np.delete(classes_nucleus_cv,idxs_test,axis=0).astype('float32')
                classes_test_onset = classes_onset_cv[idx_start:idx_end].astype('float32')
                classes_test_nucleus = classes_nucleus_cv[idx_start:idx_end].astype('float32')

                print('Calculating onset importances...')

                forest = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)

                forest.fit(dataset_train, classes_train_onset)
                results = permutation_importance(forest, dataset_test, classes_test_onset, n_repeats=5, random_state=0)

                indices_sorted = np.array(results.importances_mean).argsort()[::-1]
                importances_sorted = np.sort(np.array(results.importances_mean))[::-1]
                names_sorted = features_names[indices_sorted.tolist()]

                np.save('data/processed/' + mode + '/names_sorted_onset_eng_' + mode + '_' + str(cv) + '_' + str(it), names_sorted)
                np.save('data/processed/' + mode + '/indices_sorted_onset_eng_' + mode + '_' + str(cv) + '_' + str(it), indices_sorted)
                np.save('data/processed/' + mode + '/importances_sorted_onset_eng_' + mode + '_' + str(cv) + '_' + str(it), importances_sorted)

                print('Calculating nucleus importances...')

                forest = RandomForestClassifier(n_estimators=100,random_state=it,n_jobs=-1)

                forest.fit(dataset_train, classes_train_nucleus)
                results = permutation_importance(forest, dataset_test, classes_test_nucleus, n_repeats=5, random_state=it)

                indices_sorted = np.array(results.importances_mean).argsort()[::-1]
                importances_sorted = np.sort(np.array(results.importances_mean))[::-1]
                names_sorted = features_names[indices_sorted.tolist()]

                np.save('data/processed/' + mode + '/names_sorted_nucleus_eng_' + mode + '_' + str(cv) + '_' + str(it), names_sorted)
                np.save('data/processed/' + mode + '/indices_sorted_nucleus_eng_' + mode + '_' + str(cv) + '_' + str(it), indices_sorted)
                np.save('data/processed/' + mode + '/importances_sorted_nucleus_eng_' + mode + '_' + str(cv) + '_' + str(it), importances_sorted)
        
    elif 'class' in mode or 'sound' in mode:

        if mode=='sound':
            if not os.path.isdir('data/processed/' + mode):
                os.mkdir('data/processed/' + mode)

        np.random.seed(0)
        np.random.shuffle(classes)

        for cv in range(num_crossval):

            for it in range(num_iterations):

                dataset_cv = dataset.copy()
                classes_cv = classes.copy()

                idx_start = cv*cutoff_test
                idx_end = (cv+1)*cutoff_test

                idxs_test = np.arange(idx_start,idx_end).tolist()

                dataset_train = np.delete(dataset_cv,idxs_test,axis=0).astype('float32')
                dataset_test = dataset_cv[idx_start:idx_end].astype('float32')

                classes_train = np.delete(classes_cv,idxs_test,axis=0).astype('float32')
                classes_test = classes_cv[idx_start:idx_end].astype('float32')

                print('Calculating importances...')

                forest = RandomForestClassifier(n_estimators=100,random_state=it,n_jobs=-1)

                forest.fit(dataset_train, classes_train)
                results = permutation_importance(forest, dataset_test, classes_test, n_repeats=5, random_state=it)

                indices_sorted = np.array(results.importances_mean).argsort()[::-1]
                importances_sorted = np.sort(np.array(results.importances_mean))[::-1]
                names_sorted = features_names[indices_sorted.tolist()]

                np.save('data/processed/' + mode + '/names_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), names_sorted)
                np.save('data/processed/' + mode + '/indices_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), indices_sorted)
                np.save('data/processed/' + mode + '/importances_sorted_eng_' + mode + '_' + str(cv) + '_' + str(it), importances_sorted)
        

