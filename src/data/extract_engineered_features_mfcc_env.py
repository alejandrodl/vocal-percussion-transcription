import os
import pdb
import numpy as np

import essentia
from essentia.standard import *


class extract_features_32():
    
    def __init__(self):
        
        self.num_extra_features = 4
        self.num_mel_coeffs = 14
        self.num_mel_bands = 40
        self.num_features = 32
        
    def compute(self, audio, framesize):
        
        hopsize = 512
        audio = essentia.array(audio)
        feature_vector = np.zeros(self.num_features)
        
        # Define preprocessing functions
        
        win = Windowing(type='hann')
        spec = Spectrum(size=framesize)
        
        # Define extractors
        
        mfcc = MFCC(highFrequencyBound=22050,numberCoefficients=self.num_mel_coeffs,numberBands=self.num_mel_bands,inputSize=(framesize//2)+1)
        
        env = Envelope(applyRectification=True, attackTime=5, releaseTime=100)
        der = DerivativeSFX()
        flat = FlatnessSFX()
        tct = TCToTotal()
        
        # Extract and allocate envelope features
        
        envelope = env(audio)
        feature_vector[0], feature_vector[1] = der(envelope)
        feature_vector[2] = flat(envelope)
        feature_vector[3] = tct(envelope)

        # Extract MFCC

        _mfccs = []
        for frame in FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
            _, _mfccv = mfcc(spec(win(frame)))
            _mfccs.append(_mfccv)
            
        # Allocate MFCC
            
        for i in range(self.num_mel_coeffs):
            feature_vector[4+i] = np.mean(np.array(_mfccs)[:,i])
            feature_vector[4+self.num_mel_coeffs+i] = np.mean(np.gradient(np.array(_mfccs)[:,i]))

        return feature_vector


class extract_features_16():
    
    def __init__(self):
        
        self.num_extra_features = 4
        self.num_mel_coeffs = 12
        self.num_mel_bands = 40
        self.num_features = 16
        
    def compute(self, audio, framesize):
        
        hopsize = 512
        audio = essentia.array(audio)
        feature_vector = np.zeros(self.num_features)
        
        # Define preprocessing functions
        
        win = Windowing(type='hann')
        spec = Spectrum(size=framesize)
        
        # Define extractors
        
        mfcc = MFCC(highFrequencyBound=22050,numberCoefficients=self.num_mel_coeffs,numberBands=self.num_mel_bands,inputSize=(framesize//2)+1)
        
        env = Envelope(applyRectification=True, attackTime=5, releaseTime=100)
        der = DerivativeSFX()
        flat = FlatnessSFX()
        tct = TCToTotal()
        
        # Extract and allocate envelope features
        
        envelope = env(audio)
        feature_vector[0], feature_vector[1] = der(envelope)
        feature_vector[2] = flat(envelope)
        feature_vector[3] = tct(envelope)

        # Extract MFCC

        _mfccs = []
        for frame in FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
            _, _mfccv = mfcc(spec(win(frame)))
            _mfccs.append(_mfccv)
            
        # Allocate MFCC
            
        for i in range(self.num_mel_coeffs):
            feature_vector[4+i] = np.mean(np.array(_mfccs)[:,i])

        return feature_vector
    
    
mode = 'eng_mfcc_env'
frame_size = 1024

if not os.path.isdir('../../data/processed/' + mode):
    os.mkdir('../../data/processed/' + mode)

for part in range(28):

    if part<=9:
        list_audio = np.load('../../data/external/AVP_Dataset_Audio/Dataset_Test_0' + str(part) + '.npy', allow_pickle=True)
    else:
        list_audio = np.load('../../data/external/AVP_Dataset_Audio/Dataset_Test_' + str(part) + '.npy', allow_pickle=True)

    print(list_audio.shape)
    extractor = extract_features_32()
    features_32 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_32[n] = extractor.compute(list_audio[n],frame_size)

    print(list_audio.shape)
    extractor = extract_features_16()
    features_16 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_16[n] = extractor.compute(list_audio[n],frame_size)

    if part==0:
        print(features_32.shape)
        print(features_16.shape)

    if part<=9:
        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_32_0' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_16_0' + str(part), features_16)
    else:
        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_32_' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/test_features_avp_' + mode + '_16_' + str(part), features_16)

for part in range(28):

    if part<=9:
        list_audio = np.load('../../data/external/AVP_Dataset_Audio/Dataset_Train_0' + str(part) + '.npy', allow_pickle=True)
    else:
        list_audio = np.load('../../data/external/AVP_Dataset_Audio/Dataset_Train_' + str(part) + '.npy', allow_pickle=True)

    print(list_audio.shape)
    extractor = extract_features_32()
    features_32 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_32[n] = extractor.compute(list_audio[n],frame_size)

    print(list_audio.shape)
    extractor = extract_features_16()
    features_16 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_16[n] = extractor.compute(list_audio[n],frame_size)

    if part==0:
        print(features_32.shape)
        print(features_16.shape)

    if part<=9:
        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_32_0' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_16_0' + str(part), features_16)
    else:
        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_32_' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/train_features_avp_' + mode + '_16_' + str(part), features_16)


for part in range(20):

    if part<=9:
        list_audio = np.load('../../data/external/LVT_Dataset_Audio/Dataset_Test_0' + str(part) + '.npy', allow_pickle=True)
    else:
        list_audio = np.load('../../data/external/LVT_Dataset_Audio/Dataset_Test_' + str(part) + '.npy', allow_pickle=True)

    print(list_audio.shape)
    extractor = extract_features_32()
    features_32 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_32[n] = extractor.compute(list_audio[n],frame_size)

    print(list_audio.shape)
    extractor = extract_features_16()
    features_16 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_16[n] = extractor.compute(list_audio[n],frame_size)

    if part==0:
        print(features_32.shape)
        print(features_16.shape)

    if part<=9:
        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_32_0' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_16_0' + str(part), features_16)
    else:
        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_32_' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/test_features_lvt_' + mode + '_16_' + str(part), features_16)

for part in range(20):

    if part<=9:
        list_audio = np.load('../../data/external/LVT_Dataset_Audio/Dataset_Train_0' + str(part) + '.npy', allow_pickle=True)
    else:
        list_audio = np.load('../../data/external/LVT_Dataset_Audio/Dataset_Train_' + str(part) + '.npy', allow_pickle=True)

    print(list_audio.shape)
    extractor = extract_features_32()
    features_32 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_32[n] = extractor.compute(list_audio[n],frame_size)

    print(list_audio.shape)
    extractor = extract_features_16()
    features_16 = np.zeros((list_audio.shape[0],extractor.num_features))
    for n in range(list_audio.shape[0]):
        features_16[n] = extractor.compute(list_audio[n],frame_size)

    if part==0:
        print(features_32.shape)
        print(features_16.shape)

    if part<=9:
        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_32_0' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_16_0' + str(part), features_16)
    else:
        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_32_' + str(part), features_32)
        np.save('../../data/processed/' + mode + '/train_features_lvt_' + mode + '_16_' + str(part), features_16)


