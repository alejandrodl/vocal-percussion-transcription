import tensorflow as tf
import numpy as np



def saliency_map(model, x, label=None):
    x = tf.Variable(x, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(x, training=False)
        #input(tf.shape(pred))
        if label is None:
            class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
            loss = pred[0][class_idxs_sorted[0]]
        else:
            loss = pred[0][label]
        grads = tape.gradient(loss, x)
    return grads


def loss_gradient(model, x, label=None, loss_function=None):
    x = tf.Variable(x, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(x, training=False)
        if label is None:
            class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
            label = [class_idxs_sorted[0]]
        loss = loss_function(label, pred)
        grads = tape.gradient(loss, x)
    return grads


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

        if np.greater(onset_loss+nucleus_loss, self.best_loss):
            self.best_loss = onset_loss+nucleus_loss
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


def load_classes_sound(list_test_participants_avp, list_test_participants_lvt):
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
    
    return classes


def load_classes_syll_phon(mode, list_test_participants_avp, list_test_participants_lvt):

    if 'syllall' in mode or 'phonall' in mode:

        classes_onset = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Test_Aug_' + str(n) + '.npy')))
        classes_onset = classes_onset[1:]

        classes_nucleus = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Test_Aug_' + str(n) + '.npy')))
        classes_nucleus = classes_nucleus[1:]

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

            return classes

        else:

            return [classes_onset,classes_nucleus]

    elif 'syllred' in mode or 'phonred' in mode:

        classes_onset = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_onset = np.concatenate((classes_onset, np.load('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_' + str(n) + '.npy')))
        classes_onset = classes_onset[1:]

        classes_nucleus = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_0' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_' + str(n) + '.npy')))
                    classes_nucleus = np.concatenate((classes_nucleus, np.load('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_' + str(n) + '.npy')))
        classes_nucleus = classes_nucleus[1:]

        num_onset = np.max(classes_onset)+1
        num_nucleus = np.max(classes_nucleus)+1

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

            return classes

        else:

            return [classes_onset,classes_nucleus]

def load_classes_instrument(mode, list_test_participants_avp, list_test_participants_lvt):

    if mode=='classall':

        classes_str = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
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

    elif mode=='classred':

        classes_str = np.zeros(1)
        for n in range(28):
            if n in list_test_participants_avp:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/AVP/Classes_Test_Aug_' + str(n) + '.npy')))
        for n in range(20):
            if n in list_test_participants_lvt:
                continue
            else:
                if n<=9:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_0' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_0' + str(n) + '.npy')))
                else:
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Train_Aug_' + str(n) + '.npy')))
                    classes_str = np.concatenate((classes_str, np.load('data/interim/LVT/Classes_Test_Aug_' + str(n) + '.npy')))
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

    return classes




    

