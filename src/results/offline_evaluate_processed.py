import os
import numpy as np
import sklearn as skl
import tensorflow as tf
import tensorflow_probability as tfp

from networks_offline import *
from src.utils import *



os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

num_it = 10
percentage_train = 70
modes = ['vae','classall','classred','syllall','syllred','phonall','phonred'] # Triplet!!
clfs = ['mlp','logr','knn','rf','xgboost']

# Data parameters

frame_sizes = ['512','1024','2048']

# Network parameters

latent_dim = 32
lr = 1e-3

# MLP parameters

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

n_estimators = 300
max_depth = 10

# Gradient Boosting Trees parameters

max_depth = 10
min_child_weight = 5
colsample = 0.8
subsample = 0.9
learning_rate = 0.01
reg_lambda = 1.2
reg_alpha = 1.2
n_estimators = 300

params = {'max_depth':max_depth,
          'min_child_weight': min_child_weight,
          'learning_rate':learning_rate,
          'subsample': subsample,
          'colsample_bytree': colsample,
          'objective':'multi:softmax',
          'reg_lambda':reg_lambda,
          'reg_alpha':reg_alpha,
          'n_estimators':n_estimators,
          'tree_method':'gpu_hist',
          'gpu_id':0}

classes_also_valid = np.load('../../data/interim/AVP/Classes_Also_Valid_AVP.npy',allow_pickle=True)

accuracies = np.zeros((len(frame_sizes),28,num_it))
accuracies_also_valid = np.zeros((len(frame_sizes),28,num_it))

for mode in modes:

    if not os.path.isdir('../../results/' + mode):
        os.mkdir('../../results/' + mode)

    for a in range(len(frame_sizes)):

        frame_size = frame_sizes[a]

        for part in range(28):

            # Load and process spectrograms

            print('\n')
            print('Hyperparameters: ' + str([frame_size,kernel_height,kernel_width,part]))
            print('\n')

            if part<=9:
                classes_str = np.load('../../data/interim/AVP/Classes_Test_0' + str(part) + '.npy')
            else:
                classes_str = np.load('../../data/interim/AVP/Classes_Test_' + str(part) + '.npy')

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
                classes_str = np.load('../../data/interim/AVP/Classes_Train_Aug_0' + str(part) + '.npy')
            else:
                classes_str = np.load('../../data/interim/AVP/Classes_Train_Aug_' + str(part) + '.npy')

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

            if part<=9:
                dataset = np.load('../../data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_0' + str(part) + '_' + str(it) + '.npy')
                dataset_eval = np.load('../../data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_0' + str(part) + '_' + str(it) + '.npy')
            else:
                dataset = np.load('../../data/processed/' + mode + '/train_features_' + mode + '_' + frame_size + '_' + str(part) + '_' + str(it) + '.npy')
                dataset_eval = np.load('../../data/processed/' + mode + '/test_features_' + mode + '_' + frame_size + '_' + str(part) + '_' + str(it) + '.npy')
            
            dataset = dataset-np.mean(np.vstack((dataset,dataset_eval)))/np.std(np.vstack((dataset,dataset_eval)))
            dataset_eval = dataset_eval-np.mean(np.vstack((dataset,dataset_eval)))/np.std(np.vstack((dataset,dataset_eval)))

            np.random.seed(0)
            np.random.shuffle(dataset)

            np.random.seed(0)
            np.random.shuffle(classes)

            cutoff_train = int((percentage_train/100)*dataset.shape[0])

            dataset = dataset.astype('float32')
            classes = classes.astype('float32')
            dataset_eval = dataset_eval.astype('float32')
            classes_eval = classes_eval.astype('float32')

            for clf in clfs:

                if clf=='mlp':

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

                            model = MLP_VPT(np.max(classes)+1, latent_dim)

                            optimizer = tf.keras.optimizers.Adam(lr)
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience_early, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
                            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=patience_lr, verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

                            model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                            history = model.fit(train_features, train_classes, batch_size=batch_size, epochs=epochs, validation_data=(val_features,val_classes), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)  # , verbose=0

                            test_loss, test_acc = model.evaluate(eval_features, eval_classes, verbose=2)
                            predicted = np.argmax(model.predict(eval_features),axis=1)

                            accuracies[a,part,it] = test_acc
                            accuracies_also_valid[a,part,it] = also_valid_function(classes_eval_also_valid, predicted)
                            print('Also Valid: ' + str(accuracies_also_valid[a,part,it]))

                            np.save('../../results/Accuracies_' + mode + '_' + clf + '_' + frame_size, accuracies)
                            np.save('../../results/Accuracies_Also_Valid_' + mode + '_' + clf + '_' + frame_size, accuracies_also_valid)

                elif clf=='logr':

                        clf = skl.linear_model.LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter)
                        clf.fit(dataset, classes)

                        accuracies[a,part,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                elif clf=='knn':

                        clf = skl.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
                        clf.fit(dataset, classes)

                        accuracies[a,part,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                elif clf=='rf':

                        clf = skl.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        clf.fit(dataset, classes)

                        accuracies[a,part,it] = clf.score(dataset_eval, classes_eval)
                        accuracies_also_valid[a,part,it] = also_valid_function(classes_eval_also_valid, clf.predict(dataset_eval))

                elif clf=='xgboost':

                        model = xgb.XGBClassifier(**params)
                        model.fit(dataset, classes, eval_metric='merror')

                        accuracies[a,part,it] = sklearn.metrics.accuracy_score(classes_eval, model.predict(dataset_eval))
                        accuracies_also_valid[a,part,it] = also_valid_function(classes_eval_also_valid, model.predict(dataset_eval))
                        