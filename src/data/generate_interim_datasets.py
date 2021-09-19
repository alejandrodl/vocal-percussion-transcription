#!/usr/bin/env python
# coding: utf-8


import IPython.display as ipd
import soundfile as sf
import IPython
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.io.wavfile
import sys
import logging
import aubio
import librosa
from librosa.util import frame
import glob
import os
import time
import pdb
import random
from random import randrange
import shutil
import copy
#import rubberband
import pyrubberband as pyrb


def Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes):

    Onset_Phonemes_Labels = np.zeros(Onset_Phonemes.shape)
    for n in range(len(Onset_Phonemes)):
        if 'ts' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 0
        elif 'tʃ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 1
        elif 'tɕ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 2
        elif 'kg' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 3
        elif 'tʒ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 4
        elif 'ʡʢ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 5
        elif 'dʒ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 6
        elif 'kʃ' in Onset_Phonemes[n]:
            Onset_Phonemes_Labels[n] = 7
        elif Onset_Phonemes[n]=='t':
            Onset_Phonemes_Labels[n] = 8
        elif Onset_Phonemes[n]=='p':
            Onset_Phonemes_Labels[n] = 9
        elif Onset_Phonemes[n]=='k':
            Onset_Phonemes_Labels[n] = 10
        elif Onset_Phonemes[n]=='s':
            Onset_Phonemes_Labels[n] = 11
        elif Onset_Phonemes[n]=='!':
            Onset_Phonemes_Labels[n] = 12
            
    Nucleus_Phonemes_Labels = np.zeros(Nucleus_Phonemes.shape)
    for n in range(len(Nucleus_Phonemes)):
        if 'a' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 0
        elif 'e' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 1
        elif 'i' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 2
        elif 'o' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 3
        elif 'u' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 4
        elif 'æ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 5
        elif 'œ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 6
        elif 'ə' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 7
        elif 'ʊ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 8
        elif 'ɯ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 9
        elif 'y' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 10
        elif 'ɪ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 11
        elif 'ɐ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 12
        elif 'ʌ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 13
        elif 'h' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Labels[n] = 14
        else:
            Nucleus_Phonemes_Labels[n] = 15
               
    Onset_Phonemes_Reduced_Labels = np.zeros(Onset_Phonemes.shape)
    for n in range(len(Onset_Phonemes)):
        if 'ts' in Onset_Phonemes[n] or Onset_Phonemes[n]=='s':
            Onset_Phonemes_Reduced_Labels[n] = 0
        elif 'tʃ' in Onset_Phonemes[n] or 'tɕ' in Onset_Phonemes[n] or 'dʒ' in Onset_Phonemes[n] or 'tʒ' in Onset_Phonemes[n]:
            Onset_Phonemes_Reduced_Labels[n] = 1
        elif 'kg' in Onset_Phonemes[n] or Onset_Phonemes[n]=='k' or 'kʃ' in Onset_Phonemes[n]:
            Onset_Phonemes_Reduced_Labels[n] = 2
        elif 'ʡʢ' in Onset_Phonemes[n] or Onset_Phonemes[n]=='p':
            Onset_Phonemes_Reduced_Labels[n] = 3
        elif Onset_Phonemes[n]=='t' or Onset_Phonemes[n]=='!':
            Onset_Phonemes_Reduced_Labels[n] = 4
            
    Nucleus_Phonemes_Reduced_Labels = np.zeros(Nucleus_Phonemes.shape)
    for n in range(len(Nucleus_Phonemes)):
        if 'a' in Nucleus_Phonemes[n] or 'æ' in Nucleus_Phonemes[n] or 'ɐ' in Nucleus_Phonemes[n] or 'ʌ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 0
        elif 'e' in Nucleus_Phonemes[n] or 'œ' in Nucleus_Phonemes[n] or 'ə' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 1
        elif 'i' in Nucleus_Phonemes[n] or 'y' in Nucleus_Phonemes[n] or 'ɪ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 2
        elif 'o' in Nucleus_Phonemes[n] or 'ʊ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 3
        elif 'u' in Nucleus_Phonemes[n] or 'ɯ' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 4
        elif 'h' in Nucleus_Phonemes[n]:
            Nucleus_Phonemes_Reduced_Labels[n] = 5
        else:
            Nucleus_Phonemes_Reduced_Labels[n] = 6
            
    return Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels


    
def pitch_shift(data, sampling_rate, pitch_semitones):
    return pyrb.pitch_shift(data, sampling_rate, pitch_semitones)

def time_stretch(data, stretch_factor):
    return pyrb.time_stretch(data, 44100, stretch_factor)



frame_sizes = [1024]
num_specs = [64]
num_frames = 48

hop_size = 512
delta_bool = False



'''# Create AVP Test Dataset

print('AVP Test')

path_audio = 'data/external/AVP_Dataset/Personal'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

list_wav = list_wav[2::5]
list_csv = list_csv[2::5]

for i in range(len(list_wav)):

    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)
    Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
    Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
    Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

    Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_samples = onsets_samples.astype(int)
    
    for j in range(len(num_specs)):
        
        for w in range(len(frame_sizes)):
        
            frame_size = frame_sizes[w]
            num_spec = num_specs[j]
            
            spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

            if delta_bool:
                delta = librosa.feature.delta(spec)
                Dataset_Spec = np.concatenate((spec, delta), axis=1)
            else:
                Dataset_Spec = spec

            Onsets = np.zeros(spec.shape[0])
            location = np.floor(onsets_samples/hop_size)
            if (location.astype(int)[-1]<len(Onsets)):
                Onsets[location.astype(int)] = 1
            else:
                Onsets[location.astype(int)[:-1]] = 1

            num_onsets = int(np.sum(Onsets))
            Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

            L = len(Onsets)
            count = 0
            for n in range(L):
                if Onsets[n]==1:
                    c = 1
                    while Onsets[n+c]==0 and (n+c)<L-1:
                        c += 1
                    Spec = Dataset_Spec[n:n+c]
                    if c<num_frames:
                        Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                    elif c>=num_frames:
                        Spec = Spec[:num_frames]
                    Spec_Matrix[count] = Spec.T
                    count += 1
                    
            list_num = [Spec_Matrix.shape[0],len(Classes),len(Onset_Phonemes_Labels),len(Nucleus_Phonemes_Labels),len(Onset_Phonemes_Reduced_Labels),len(Nucleus_Phonemes_Reduced_Labels)]
            if list_num.count(list_num[0])!=len(list_num):
                print(list_num)
                print(list_wav[i])

            if i<=9:
                np.save('data/interim/AVP/Dataset_Test_0' + str(i), Spec_Matrix)
                np.save('data/interim/AVP/Classes_Test_0' + str(i), Classes)
                np.save('data/interim/AVP/Syll_Onset_Test_0' + str(i), Onset_Phonemes_Labels)
                np.save('data/interim/AVP/Syll_Nucleus_Test_0' + str(i), Nucleus_Phonemes_Labels)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Test_0' + str(i), Onset_Phonemes_Reduced_Labels)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Test_0' + str(i), Nucleus_Phonemes_Reduced_Labels)
            else:
                np.save('data/interim/AVP/Dataset_Test_' + str(i), Spec_Matrix)
                np.save('data/interim/AVP/Classes_Test_' + str(i), Classes)
                np.save('data/interim/AVP/Syll_Onset_Test_' + str(i), Onset_Phonemes_Labels)
                np.save('data/interim/AVP/Syll_Nucleus_Test_' + str(i), Nucleus_Phonemes_Labels)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Test_' + str(i), Onset_Phonemes_Reduced_Labels)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Test_' + str(i), Nucleus_Phonemes_Reduced_Labels)




# Create AVP Test Aug Dataset

print('AVP Test Aug')

pitch = [-1,1]
times = [0.85,1.15]

path_audio = 'data/external/AVP_Dataset/Personal'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

list_wav = list_wav[2::5]
list_csv = list_csv[2::5]

for i in range(len(list_wav)):

    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)
    Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
    
    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)
    
    for j in range(len(num_specs)):
        
        for w in range(len(frame_sizes)):
        
            frame_size = frame_sizes[w]
            num_spec = num_specs[j]
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)
            
            for k in range(10):

                Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
                Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

                Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

                kn = np.random.randint(0,2)
                pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                st = np.random.uniform(low=0.8, high=1.2, size=None)

                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
            
                spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

                if delta_bool:
                    delta = librosa.feature.delta(spec)
                    Dataset_Spec = np.concatenate((spec, delta), axis=1)
                else:
                    Dataset_Spec = spec

                Onsets = np.zeros(spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                if num_onsets!=len(Classes):
                    raise('num_onsets==len(Classes)')
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while Onsets[n+c]==0 and (n+c)<L-1:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1
                        
                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                Classes_All = np.concatenate((Classes_All,Classes))
                Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))
                
                list_num = [Spec_Matrix_All.shape[0],len(Classes_All),len(Onset_Phonemes_Labels_All),len(Nucleus_Phonemes_Labels_All),len(Onset_Phonemes_Reduced_Labels_All),len(Nucleus_Phonemes_Reduced_Labels_All)]
                if list_num.count(list_num[0])!=len(list_num):
                    print(list_num)
                    print(list_wav[i])
            
            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:]
            
            if i<=9:
                np.save('data/interim/AVP/Dataset_Test_Aug_0' + str(i), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Test_Aug_0' + str(i), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Test_Aug_0' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Test_Aug_0' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_0' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_0' + str(i), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/AVP/Dataset_Test_Aug_' + str(i), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Test_Aug_' + str(i), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Test_Aug_' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Test_Aug_' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Test_Aug_' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Test_Aug_' + str(i), Nucleus_Phonemes_Reduced_Labels_All)





# Create Train Dataset

print('AVP Train')

fs = 44100

path_audio = 'data/external/AVP_Dataset/Personal'

list_wav_all = []
list_csv_all = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav_all.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv_all.append(os.path.join(path, filename))

list_wav_all = sorted(list_wav_all)
list_csv_all = sorted(list_csv_all)

list_wav_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

list_wav = list_wav_all[::5] + list_wav_all[1::5] + list_wav_all[3::5] + list_wav_all[4::5]
list_csv = list_csv_all[::5] + list_csv_all[1::5] + list_csv_all[3::5] + list_csv_all[4::5]

list_wav_all = sorted(list_wav)
list_csv_all = sorted(list_csv)

list_wav_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

for j in range(len(num_specs)):

    for k in range(len(frame_sizes)):

        frame_size = frame_sizes[k]
        num_spec = num_specs[j]
        
        for part in range(28):
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)

            for i in range(4):

                onsets = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=0)
                Classes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=1, dtype=np.unicode_)

                audio, fs = librosa.load(list_wav_all[4*part+i], sr=44100)
                audio_ref = audio/np.max(abs(audio))

                onsets_samples = onsets*fs
                onsets_ref = onsets_samples.astype(int)
                
                for k in range(1):

                    Classes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=1, dtype=np.unicode_)
                    Onset_Phonemes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=2, dtype=np.unicode_)
                    Nucleus_Phonemes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=3, dtype=np.unicode_)

                    Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

                    if k==0:
                        audio = audio_ref.copy()
                        onsets = onsets_ref.copy()

                    spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(4096))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T
                    
                    if delta_bool:
                        delta = librosa.feature.delta(spec)
                        Dataset_Spec = np.concatenate((spec, delta), axis=1)
                    else:
                        Dataset_Spec = spec

                    Onsets = np.zeros(Dataset_Spec.shape[0])
                    location = np.floor(onsets/hop_size)
                    if (location.astype(int)[-1]<len(Onsets)):
                        Onsets[location.astype(int)] = 1
                    else:
                        Onsets[location.astype(int)[:-1]] = 1

                    if Onsets[len(Onsets)-1]==1:
                        Classes = Classes[:-1]
                        Onsets[len(Onsets)-1] = 0
                        print(len(Classes))
                        print(int(np.sum(Onsets)))

                    num_onsets = int(np.sum(Onsets))
                    if num_onsets!=len(Classes):
                        raise('num_onsets!=len(Classes)')
                    Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                    L = len(Onsets)
                    count = 0
                    for n in range(L):
                        if Onsets[n]==1:
                            c = 1
                            while Onsets[n+c]==0 and (n+c)<L-1:
                                c += 1
                            Spec = Dataset_Spec[n:n+c]
                            if c<num_frames:
                                Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                            elif c>=num_frames:
                                Spec = Spec[:num_frames]
                            Spec_Matrix[count] = Spec.T
                            count += 1

                    if Spec_Matrix.shape[0]==Classes.shape[0]:
                        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                        Classes_All = np.concatenate((Classes_All,Classes))
                        Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                        Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                        Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                        Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))
                    else:
                        print('Cuidao')
                        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                        Classes_All = np.concatenate((Classes_All,Classes))
                        Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                        Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                        Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                        Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:]        

            if part<=9:
                np.save('data/interim/AVP/Dataset_Train_0' + str(part), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Train_0' + str(part), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Train_0' + str(part), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Train_0' + str(part), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Train_0' + str(part), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Train_0' + str(part), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/AVP/Dataset_Train_' + str(part), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Train_' + str(part), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Train_' + str(part), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Train_' + str(part), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Train_' + str(part), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Train_' + str(part), Nucleus_Phonemes_Reduced_Labels_All)




# Create Train Aug Dataset

print('AVP Train Aug')

fs = 44100

path_audio = 'data/external/AVP_Dataset/Personal'

list_wav_all = []
list_csv_all = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('.wav'):
            list_wav_all.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv_all.append(os.path.join(path, filename))

list_wav_all = sorted(list_wav_all)
list_csv_all = sorted(list_csv_all)

list_wav_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

list_wav = list_wav_all[::5] + list_wav_all[1::5] + list_wav_all[3::5] + list_wav_all[4::5]
list_csv = list_csv_all[::5] + list_csv_all[1::5] + list_csv_all[3::5] + list_csv_all[4::5]

list_wav_all = sorted(list_wav)
list_csv_all = sorted(list_csv)

list_wav_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
list_csv_all.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

for j in range(len(num_specs)):

    for w in range(len(frame_sizes)):

        frame_size = frame_sizes[w]
        num_spec = num_specs[j]
        
        for part in range(28):
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)

            for i in range(4):

                onsets = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=0)
                Classes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=1, dtype=np.unicode_)

                audio, fs = librosa.load(list_wav_all[4*part+i], sr=44100)
                audio_ref = audio/np.max(abs(audio))

                onsets_samples = onsets*fs
                onsets_ref = onsets_samples.astype(int)
                
                for k in range(10):

                    Classes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=1, dtype=np.unicode_)
                    Onset_Phonemes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=2, dtype=np.unicode_)
                    Nucleus_Phonemes = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=3, dtype=np.unicode_)

                    Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

                    kn = np.random.randint(0,2)
                    pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                    st = np.random.uniform(low=0.8, high=1.2, size=None)

                    if kn==0:
                        audio = pitch_shift(audio_ref, fs, pt)
                        audio = time_stretch(audio, st)
                        onsets = onsets_ref/st
                        onsets = onsets.astype(int)
                    elif kn==1:
                        audio = time_stretch(audio_ref, st)
                        audio = pitch_shift(audio, fs, pt)
                        onsets = onsets_ref/st
                        onsets = onsets.astype(int)

                    spec = librosa.feature.melspectrogram(np.concatenate((audio,np.zeros(4096))), sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T
                    
                    if delta_bool:
                        delta = librosa.feature.delta(spec)
                        Dataset_Spec = np.concatenate((spec, delta), axis=1)
                    else:
                        Dataset_Spec = spec

                    Onsets = np.zeros(Dataset_Spec.shape[0])
                    location = np.floor(onsets/hop_size)
                    if (location.astype(int)[-1]<len(Onsets)):
                        Onsets[location.astype(int)] = 1
                    else:
                        Onsets[location.astype(int)[:-1]] = 1

                    if Onsets[len(Onsets)-1]==1:
                        Classes = Classes[:-1]
                        Onsets[len(Onsets)-1] = 0
                        print(len(Classes))
                        print(int(np.sum(Onsets)))

                    num_onsets = int(np.sum(Onsets))
                    if num_onsets!=len(Classes):
                        raise('num_onsets!=len(Classes)')
                    Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                    L = len(Onsets)
                    count = 0
                    for n in range(L):
                        if Onsets[n]==1:
                            c = 1
                            while Onsets[n+c]==0 and (n+c)<L-1:
                                c += 1
                            Spec = Dataset_Spec[n:n+c]
                            if c<num_frames:
                                Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                            elif c>=num_frames:
                                Spec = Spec[:num_frames]
                            Spec_Matrix[count] = Spec.T
                            count += 1

                    list_num = [Spec_Matrix_All.shape[0],len(Classes_All),len(Onset_Phonemes_Labels_All),len(Nucleus_Phonemes_Labels_All),len(Onset_Phonemes_Reduced_Labels_All),len(Nucleus_Phonemes_Reduced_Labels_All)]
                    if list_num.count(list_num[0])==len(list_num):
                        Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                        Classes_All = np.concatenate((Classes_All,Classes))
                        Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                        Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                        Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                        Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))
                    else:
                        print(list_num)
                        print(list_wav[i])

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:]       

            if part<=9:
                np.save('data/interim/AVP/Dataset_Train_Aug_0' + str(part), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Train_Aug_0' + str(part), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Train_Aug_0' + str(part), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Train_Aug_0' + str(part), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_0' + str(part), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_0' + str(part), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/AVP/Dataset_Train_Aug_' + str(part), Spec_Matrix_All)
                np.save('data/interim/AVP/Classes_Train_Aug_' + str(part), Classes_All)
                np.save('data/interim/AVP/Syll_Onset_Train_Aug_' + str(part), Onset_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Train_Aug_' + str(part), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/AVP/Syll_Onset_Reduced_Train_Aug_' + str(part), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/AVP/Syll_Nucleus_Reduced_Train_Aug_' + str(part), Nucleus_Phonemes_Reduced_Labels_All)'''




# Create LVT Train Dataset

print('LVT Train')

path_audio = 'data/external/LVT_Dataset'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('3.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav = list_wav[:20]
list_csv = list_csv[:20]

for j in range(len(num_specs)):
    
    for w in range(len(frame_sizes)):

        frame_size = frame_sizes[w]
        num_spec = num_specs[j]

        for i in range(len(list_wav)):
            
            onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

            audio, fs = librosa.load(list_wav[i], sr=44100)
            audio_ref = audio/np.max(abs(audio))

            onsets_samples = onsets*fs
            onsets_ref = onsets_samples.astype(int)
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)
            
            for k in range(1):

                Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
                Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

                Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)
                
                audio = audio_ref
                onsets = onsets_ref

                Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

                Onsets = np.zeros(Dataset_Spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while Onsets[n+c]==0 and (n+c)<L-1:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                Classes_All = np.concatenate((Classes_All,Classes))
                Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:] 

            if i<=9:
                np.save('data/interim/LVT/Dataset_Train_0' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Train_0' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Train_0' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Train_0' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Train_0' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Train_0' + str(i), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/LVT/Dataset_Train_' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Train_' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Train_' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Train_' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Train_' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Train_' + str(i), Nucleus_Phonemes_Reduced_Labels_All)








# Create LVT Train Aug Dataset

print('LVT Train Aug')

path_audio = 'data/external/LVT_Dataset'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('3.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav = list_wav[:20]
list_csv = list_csv[:20]

for j in range(len(num_specs)):
    
    for w in range(len(frame_sizes)):

        frame_size = frame_sizes[w]
        num_spec = num_specs[j]

        for i in range(len(list_wav)):
            
            onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

            audio, fs = librosa.load(list_wav[i], sr=44100)
            audio_ref = audio/np.max(abs(audio))

            onsets_samples = onsets*fs
            onsets_ref = onsets_samples.astype(int)
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)
            
            for k in range(10):

                Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
                Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

                Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)
                
                kn = np.random.randint(0,2)
                pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                st = np.random.uniform(low=0.8, high=1.2, size=None)

                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)

                Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

                Onsets = np.zeros(Dataset_Spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while Onsets[n+c]==0 and (n+c)<L-1:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                Classes_All = np.concatenate((Classes_All,Classes))
                Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:] 

            if i<=9:
                np.save('data/interim/LVT/Dataset_Train_Aug_0' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Train_Aug_0' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Train_Aug_0' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Train_Aug_0' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_0' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_0' + str(i), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/LVT/Dataset_Train_Aug_' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Train_Aug_' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Train_Aug_' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Train_Aug_' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Train_Aug_' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Train_Aug_' + str(i), Nucleus_Phonemes_Reduced_Labels_All)







# Create LVT Test Dataset

print('LVT Test')

path_audio = 'data/external/LVT_Dataset'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('3.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav = list_wav[20:]
list_csv = list_csv[20:]

for j in range(len(num_specs)):
    
    for w in range(len(frame_sizes)):

        frame_size = frame_sizes[w]
        num_spec = num_specs[j]

        for i in range(len(list_wav)):
            
            onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

            audio, fs = librosa.load(list_wav[i], sr=44100)
            audio_ref = audio/np.max(abs(audio))

            onsets_samples = onsets*fs
            onsets_ref = onsets_samples.astype(int)
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)
            
            for k in range(1):

                Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
                Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

                Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

                audio = audio_ref
                onsets = onsets_ref

                Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

                Onsets = np.zeros(Dataset_Spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while Onsets[n+c]==0 and (n+c)<L-1:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                Classes_All = np.concatenate((Classes_All,Classes))
                Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:] 

            if i<=9:
                np.save('data/interim/LVT/Dataset_Test_0' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Test_0' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Test_0' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Test_0' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Test_0' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Test_0' + str(i), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/LVT/Dataset_Test_' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Test_' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Test_' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Test_' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Test_' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Test_' + str(i), Nucleus_Phonemes_Reduced_Labels_All)





# Create LVT Test Aug Dataset

print('LVT Test Aug')

path_audio = 'data/external/LVT_Dataset'

list_wav = []
list_csv = []

for path, subdirs, files in os.walk(path_audio):
    for filename in files:
        if filename.endswith('3.wav'):
            list_wav.append(os.path.join(path, filename))
        if filename.endswith('.csv'):
            list_csv.append(os.path.join(path, filename))

list_wav = sorted(list_wav)
list_csv = sorted(list_csv)

list_wav = list_wav[20:]
list_csv = list_csv[20:]

for j in range(len(num_specs)):
    
    for w in range(len(frame_sizes)):

        frame_size = frame_sizes[w]
        num_spec = num_specs[j]

        for i in range(len(list_wav)):
            
            onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

            audio, fs = librosa.load(list_wav[i], sr=44100)
            audio_ref = audio/np.max(abs(audio))

            onsets_samples = onsets*fs
            onsets_ref = onsets_samples.astype(int)
            
            Spec_Matrix_All = np.zeros((1,num_spec,num_frames))
            Classes_All = np.zeros(1)
            Onset_Phonemes_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Labels_All = np.zeros(1)
            Onset_Phonemes_Reduced_Labels_All = np.zeros(1)
            Nucleus_Phonemes_Reduced_Labels_All = np.zeros(1)
            
            for k in range(10):

                Classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                Onset_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=2, dtype=np.unicode_)
                Nucleus_Phonemes = np.loadtxt(list_csv[i], delimiter=',', usecols=3, dtype=np.unicode_)

                Onset_Phonemes_Labels, Nucleus_Phonemes_Labels, Onset_Phonemes_Reduced_Labels, Nucleus_Phonemes_Reduced_Labels = Create_Phoneme_Labels(Onset_Phonemes, Nucleus_Phonemes)

                kn = np.random.randint(0,2)
                pt = np.random.uniform(low=-1.5, high=1.5, size=None)
                st = np.random.uniform(low=0.8, high=1.2, size=None)

                if kn==0:
                    audio = pitch_shift(audio_ref, fs, pt)
                    audio = time_stretch(audio, st)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)
                elif kn==1:
                    audio = time_stretch(audio_ref, st)
                    audio = pitch_shift(audio, fs, pt)
                    onsets = onsets_ref/st
                    onsets = onsets.astype(int)

                Dataset_Spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=frame_size, hop_length=hop_size, n_mels=num_spec, power=1.0).T

                Onsets = np.zeros(Dataset_Spec.shape[0])
                location = np.floor(onsets/hop_size)
                if (location.astype(int)[-1]<len(Onsets)):
                    Onsets[location.astype(int)] = 1
                else:
                    Onsets[location.astype(int)[:-1]] = 1

                num_onsets = int(np.sum(Onsets))
                Spec_Matrix = np.zeros((num_onsets,num_spec,num_frames))

                L = len(Onsets)
                count = 0
                for n in range(L):
                    if Onsets[n]==1:
                        c = 1
                        while Onsets[n+c]==0 and (n+c)<L-1:
                            c += 1
                        Spec = Dataset_Spec[n:n+c]
                        if c<num_frames:
                            Spec = np.concatenate((Spec,np.zeros((num_frames-c,num_spec))))
                        elif c>=num_frames:
                            Spec = Spec[:num_frames]
                        Spec_Matrix[count] = Spec.T
                        count += 1

                Spec_Matrix_All = np.vstack((Spec_Matrix_All,Spec_Matrix))
                Classes_All = np.concatenate((Classes_All,Classes))
                Onset_Phonemes_Labels_All = np.concatenate((Onset_Phonemes_Labels_All,Onset_Phonemes_Labels))
                Nucleus_Phonemes_Labels_All = np.concatenate((Nucleus_Phonemes_Labels_All,Nucleus_Phonemes_Labels))
                Onset_Phonemes_Reduced_Labels_All = np.concatenate((Onset_Phonemes_Reduced_Labels_All,Onset_Phonemes_Reduced_Labels))
                Nucleus_Phonemes_Reduced_Labels_All = np.concatenate((Nucleus_Phonemes_Reduced_Labels_All,Nucleus_Phonemes_Reduced_Labels))

            Spec_Matrix_All = Spec_Matrix_All[1:]
            Classes_All = Classes_All[1:]
            Onset_Phonemes_Labels_All = Onset_Phonemes_Labels_All[1:]
            Nucleus_Phonemes_Labels_All = Nucleus_Phonemes_Labels_All[1:]
            Onset_Phonemes_Reduced_Labels_All = Onset_Phonemes_Reduced_Labels_All[1:]
            Nucleus_Phonemes_Reduced_Labels_All = Nucleus_Phonemes_Reduced_Labels_All[1:] 

            if i<=9:
                np.save('data/interim/LVT/Dataset_Test_Aug_0' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Test_Aug_0' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Test_Aug_0' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Test_Aug_0' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_0' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_0' + str(i), Nucleus_Phonemes_Reduced_Labels_All)
            else:
                np.save('data/interim/LVT/Dataset_Test_Aug_' + str(i), Spec_Matrix_All)
                np.save('data/interim/LVT/Classes_Test_Aug_' + str(i), Classes_All)
                np.save('data/interim/LVT/Syll_Onset_Test_Aug_' + str(i), Onset_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Test_Aug_' + str(i), Nucleus_Phonemes_Labels_All)
                np.save('data/interim/LVT/Syll_Onset_Reduced_Test_Aug_' + str(i), Onset_Phonemes_Reduced_Labels_All)
                np.save('data/interim/LVT/Syll_Nucleus_Reduced_Test_Aug_' + str(i), Nucleus_Phonemes_Reduced_Labels_All)


