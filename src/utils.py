import os
import random
import numpy as np
import scipy as sp
from random import randrange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import tensorflow as tf
import tensorflow as tf
import tensorflow_probability as tfp



def also_valid_function(classes_also_valid, predicted):
    c = 0
    for n in range(len(classes_also_valid)):
        if predicted[n] in classes_also_valid[n]:
            c += 1
    accuracy = c/len(classes_also_valid)
    return accuracy
    

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('reconstructions/images_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    #plt.show()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_loss_cnn(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.
    
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        dist = (output2 - output1).pow(2).sum(1)
        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
                
                
def Siamese_To_Triplet(Siamese_Dataset, Siamese_Classes):

    indices_positive = np.where(Siamese_Classes==1)[0]
    indices_negative = np.where(Siamese_Classes==0)[0]

    Siamese_Dataset_Positive = Siamese_Dataset[indices_positive]

    size_half = Siamese_Dataset_Positive.shape[0]
    Triplet_Dataset = np.zeros((int(2*size_half), 96, 64))

    for j in range(2):

        for k in range(size_half):

            Triplet_Dataset[int(j*size_half)+k][:64] = Siamese_Dataset_Positive[k]
            Triplet_Dataset[int(j*size_half)+k][64:] = Siamese_Dataset[np.random.choice(indices_negative)][32:]

    return Triplet_Dataset

    
def save_top_params(list_best_params, list_params, n_top):
    if len(list_best_params)<n_top:
        list_best_params.append(list_params)
        list_best_params = sorted(list_best_params, key=lambda x: x[-1])
        list_best_params = list_best_params[::-1]
    elif len(list_best_params)==n_top:
        if list_best_params[-1][-1]<list_params[-1]:
            list_best_params[-1] = list_params
            list_best_params = sorted(list_best_params, key=lambda x: x[-1])
            list_best_params = list_best_params[::-1]
    return list_best_params
    
    
class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            #print('EarlyStopping counter: ' + str(self.counter) + ' out of ' + str(self.patience))
            #print('\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        #if self.verbose:
            #print('Validation loss decreased (' + str(self.val_loss_min) + ' --> ' + str(val_loss) + ').  Saving model ...')
        self.val_loss_min = val_loss
        
        
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)
    
    
def Prototypical_Batch_Sampler(data, classes, num_class, num_support, num_query):
    
    samples = np.zeros((num_class,int(num_support+num_query),data.size()[1],data.size()[2],data.size()[3]))
    K = np.random.choice(np.unique(classes),num_class,replace=False)
    
    for n in range(len(K)):
        cls = K[n]
        data_class = data[classes==cls]
        perm = np.random.permutation(data_class)
        samples_class = perm[:(num_support+num_query)]
        samples[n] = samples_class
        
    #samples = np.array(samples).astype(None)
    samples = torch.from_numpy(samples).float()
    #samples = samples.permute(0,1,4,2,3)
    
    return samples


def euclidean_dist(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)

    return torch.pow(x-y,2).sum(2)


def loss_ae(recon_x, x, mse_loss=True):
    
    recon_x = recon_x.squeeze(1)
    
    if mse_loss:
        mse = nn.MSELoss(reduction='mean')
        REC = mse(recon_x, x)
    else:
        bce = nn.BCELoss(reduction='mean')
        REC = bce(recon_x, x)
    
    return REC


'''def loss_vae(recon_x, x, mu, logvar, norm, mse_loss=True):
    
    recon_x = recon_x.squeeze(1)
    
    if mse_loss:
        mse = nn.MSELoss(reduction='mean')
        REC = mse(recon_x, x)
    else:
        bce = nn.BCELoss(reduction='mean')
        REC = bce(recon_x, x)

    KLD = norm*torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)/x.size()[0]
    
    return REC + KLD, REC, KLD'''


'''def loss_vae(recon_x, x, mu, logvar, mse_loss=True):
    
    recon_x = recon_x.squeeze(1)
    
    if mse_loss:
        mse = nn.MSELoss(reduction='mean')
        REC = mse(recon_x, x)
    else:
        bce = nn.BCELoss(reduction='mean')
        REC = bce(recon_x, x)

    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)/x.size()[0]
    
    return REC + KLD, REC, KLD'''

def loss_vae(recon_x, x, mu, logvar, mse_loss=True):
    
    recon_x = recon_x.squeeze(1)
    
    if mse_loss:
        mse = nn.MSELoss(reduction='mean')
        REC = torch.sqrt((recon_x-x).pow(2).mean())
    else:
        bce = nn.BCELoss(reduction='mean')
        REC = bce(recon_x, x)

    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)/x.size()[0]
    
    return REC + KLD, REC, KLD


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
    
    