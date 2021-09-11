import os
import sys
import pdb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import xgboost as xgb
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from warnings import simplefilter
simplefilter(action='ignore')

#sys.path.insert(0, os.path.abspath(__file__+"/src/"))
#from src.utils import *

#sys.path.append('/homes/adl30/vocal-percussion-transcription')
#from src.utils import *



class SLP_Processed(tf.keras.Model):

    def __init__(self, num_labels, input_length):
        super(SLP_Processed, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_labels, activation='softmax')
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out

class MLP_Processed(tf.keras.Model):

    def __init__(self, num_labels, input_length):
        super(MLP_Processed, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_length, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(12),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax')
            ]
        )

    def call(self, x):
        out = self.encoder(x)
        return out

def also_valid_function(classes_also_valid, predicted):
    c = 0
    for n in range(len(classes_also_valid)):
        if predicted[n] in classes_also_valid[n]:
            c += 1
    accuracy = c/len(classes_also_valid)
    return accuracy

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

num_it = 1
percentage_train = 70
modes = ['eng_mfcc_env','eng_all_classall','eng_all_classred','eng_all_syllall','eng_all_syllred','eng_all_phonall','eng_all_phonred',
         'vae','classall','classred','syllall','syllred','phonall','phonred'] # Triplet!!
#clfs = ['slp','mlp','logr','knn','rf','xgboost']
clfs = ['xgboost']

# Data parameters

frame_sizes = ['1024']

# Network parameters

latent_dim = 32

# MLP parameters

lr = 2*1e-3
epochs = 10000
patience_lr = 5
patience_early = 10
batch_size = 16

# Logistic Regression parameters

tol = 1e-4
reg_str = 1.0
solver = 'lbfgs'
max_iter = 100

# KNN parameters

n_neighbors = 5

# Random Forests parameters

n_estimators = 100

# Gradient Boosting Trees parameters

#max_depth = 10
#min_child_weight = 5
#colsample = 0.8
#subsample = 0.9
#learning_rate = 0.02
#reg_lambda = 1.2
#reg_alpha = 1.2
#n_estimators = 500

max_depth = 7
min_child_weight = 5
colsample = 1
subsample = 1
learning_rate = 0.15
reg_lambda = 1
reg_alpha = 0
n_estimators = 1000

params = {'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'objective': 'multi:softmax',
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'n_estimators': n_estimators} # 'seed': it, 'random_state': it, 'tree_method': 'gpu_hist', 'gpu_id': 0

classes_also_valid = np.load('data/interim/AVP/Classes_Also_Valid_AVP.npy',allow_pickle=True)

accuracies = np.zeros((len(frame_sizes),28,len(clfs),num_it))
accuracies_also_valid = np.zeros((len(frame_sizes),28,len(clfs),num_it))

for mode in modes:

    if not os.path.isdir('results/' + mode):
        os.mkdir('results/' + mode)

    for a in range(len(frame_sizes)):

        frame_size = frame_sizes[a]

        for part in range(28):

            # Load and process spectrograms

            print('\n')
            print('Hyperparameters: ' + str([frame_size,part]))
            print('\n')

            if part<=9:
                classes_str = np.load('data/interim/AVP/Classes_Test_0' + str(part) + '.npy')
            else:
                classes_str = np.load('data/interim/AVP/Classes_Test_' + str(part) + '.npy')

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

            classes_eval_also_valid = classes_also_valid[part]

            if part<=9:
                classes_str = np.load('data/interim/AVP/Classes_Train_Aug_0' + str(part) + '.npy')
            else:
                classes_str = np.load('data/interim/AVP/Classes_Train_Aug_' + str(part) + '.npy')

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

                if part<=9:
                    dataset = np.load('data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_0' + str(part) + '.npy')
                    dataset_eval = np.load('data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_0' + str(part) + '.npy')
                else:
                    dataset = np.load('data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_' + str(part) + '.npy')
                    dataset_eval = np.load('data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_' + str(part) + '.npy')

            else:

                if part<=9:
                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_' + mode[:7] + '_' + frame_size + '_0' + str(part) + '.npy')
                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_' + mode[:7] + '_' + frame_size + '_0' + str(part) + '.npy')
                else:
                    dataset = np.load('data/processed/' + mode[:7] + '/train_features_' + mode[:7] + '_' + frame_size + '_' + str(part) + '.npy')
                    dataset_eval = np.load('data/processed/' + mode[:7] + '/test_features_' + mode[:7] + '_' + frame_size + '_' + str(part) + '.npy')

                if 'phon' not in mode:

                    indices_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_eng_' + mode[8:] + '.npy')[:latent_dim]
                    indices_selected = indices_selected.tolist()
                    
                    dataset = dataset[:,indices_selected]
                    dataset_eval = dataset_eval[:,indices_selected]

                else:

                    indices_onset_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_onset_eng_' + mode[8:] + '.npy')
                    indices_onset_selected = indices_onset_selected.tolist()

                    indices_nucleus_selected = np.load('data/processed/' + mode[8:] + '/indices_sorted_nucleus_eng_' + mode[8:] + '.npy')
                    indices_nucleus_selected = indices_nucleus_selected.tolist()

                    cutoff = 32
                    indices_selected = []
                    while len(indices_selected)!=latent_dim:
                        indices_selected = list(set(indices_onset_selected[:cutoff])&set(indices_nucleus_selected[:cutoff]))
                        cutoff += 1
                    
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

                if clf=='slp' or clf=='mlp':

                    train_classes = classes[:cutoff_train]
                    val_classes = classes[cutoff_train:]
                    eval_classes = classes_eval

                    train_features = dataset[:cutoff_train]
                    val_features = dataset[cutoff_train:]
                    eval_features = dataset_eval

                    train_features = np.expand_dims(train_features,axis=-1)
                    val_features = np.expand_dims(val_features,axis=-1)
                    eval_features = np.expand_dims(eval_features,axis=-1)

                    for it in range(num_it):
                        
                        with tf.device(gpu_name):

                            if clf=='slp':
                                model = SLP_Processed(np.max(classes)+1, latent_dim)
                            else:
                                model = MLP_Processed(np.max(classes)+1, latent_dim)

                            optimizer = tf.keras.optimizers.Adam(lr)
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience_early, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
                            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=patience_lr, verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

                            model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                            history = model.fit(train_features, train_classes, batch_size=batch_size, epochs=epochs, validation_data=(val_features,val_classes), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)  # , verbose=0

                            test_loss, test_acc = model.evaluate(eval_features, eval_classes, verbose=2)
                            predicted = np.argmax(model.predict(eval_features),axis=1)

                            accuracies[a,part,b,it] = test_acc
                            accuracies_also_valid[a,part,b,it] = also_valid_function(classes_eval_also_valid, predicted)
                            print('Also Valid: ' + str(accuracies_also_valid[a,part,b,it]))

                            np.save('results/Accuracies_' + mode + '_' + clf + '_' + frame_size, accuracies)
                            np.save('results/Accuracies_Also_Valid_' + mode + '_' + clf + '_' + frame_size, accuracies_also_valid)

                elif clf=='logr':

                    for it in range(num_it):

                        clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter)
                        clf.fit(dataset, classes)

                        accuracies[a,part,b,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,b,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                        print(accuracies[a,part,b,it])
                        print(accuracies_also_valid[a,part,b,it])

                elif clf=='knn':

                    for it in range(num_it):

                        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                        clf.fit(dataset, classes)

                        accuracies[a,part,b,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,b,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                        print(accuracies[a,part,b,it])
                        print(accuracies_also_valid[a,part,b,it])

                elif clf=='rf':

                    for it in range(num_it):

                        clf = RandomForestClassifier(n_estimators=n_estimators)
                        clf.fit(dataset, classes)

                        accuracies[a,part,b,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,b,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                        print(accuracies[a,part,b,it])
                        print(accuracies_also_valid[a,part,b,it])

                elif clf=='xgboost':

                    for it in range(num_it):

                        model = xgb.XGBClassifier(**params)
                        model.fit(dataset, classes, eval_metric='merror')

                        accuracies[a,part,b,it] = accuracy_score(classes_eval, model.predict(dataset_eval))
                        accuracies_also_valid[a,part,b,it] = also_valid_function(classes_eval_also_valid, model.predict(dataset_eval))

                        print(accuracies[a,part,b,it])
                        print(accuracies_also_valid[a,part,b,it])

    np.save('results/' + mode + '/accuracies', accuracies)
    np.save('results/' + mode + '/accuracies_also_valid', accuracies_also_valid)
                        