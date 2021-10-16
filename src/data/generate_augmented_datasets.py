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



# Create Train Aug Dataset Audio

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
  
for part in range(28):

    audios_all = []

    for i in range(4):

        onsets = np.loadtxt(list_csv_all[4*part+i], delimiter=',', usecols=0)

        audio, fs = librosa.load(list_wav_all[4*part+i], sr=44100)
        audio_ref = audio/np.max(abs(audio))

        onsets_samples = onsets*fs
        onsets_ref = onsets_samples.astype(int)
        
        for k in range(10):

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

            audio = np.concatenate((audio,np.zeros(2048)))
            
            if i==0:
                class_str = 'HHC'
            elif i==1:
                class_str = 'HHO'
            elif i==2:
                class_str = 'Kick'
            elif i==3:
                class_str = 'Snare'

            if part<=9:
                sf.write('data/external/AVP_Augmented/Dataset_Train_' + class_str + '_0' + str(part) + '_' + str(k) + '.wav', audio, 44100)
                np.save('data/external/AVP_Augmented/Onsets_Train_' + class_str + '_0' + str(part) + '_' + str(k), onsets)
            else:
                sf.write('data/external/AVP_Augmented/Dataset_Train_' + class_str + '_' + str(part) + '_' + str(k) + '.wav', audio, 44100)
                np.save('data/external/AVP_Augmented/Onsets_Train_' + class_str + '_' + str(part) + '_' + str(k), onsets)



# Create Train Aug Dataset Audio

fs = 44100

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

indices_kick = [0,2,5,7,10,12,15,17]
indices_snare = [4,9,14,19]
indices_hihat = [1,3,6,8,11,13,16,18]

for i in range(len(list_wav)):
    
    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)
    
    for k in range(10):

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

        audio = np.concatenate((audio,np.zeros(2048)))

        if i<=9:
            sf.write('data/external/LVT_Audio/Dataset_Train_0' + str(i) + '_' + str(k) + '.wav', audio, 44100)
            np.save('data/external/LVT_Audio/Onsets_Train_0' + str(i) + '_' + str(k), onsets)
        else:
            sf.write('data/external/LVT_Audio/Dataset_Train_' + str(i) + '_' + str(k) + '.wav', audio, 44100)
            np.save('data/external/LVT_Audio/Onsets_Train_' + str(i) + '_' + str(k), onsets)






# Create Train Dataset Audio

fs = 44100

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

indices_kick = [0,2,5,7,10,12,15,17]
indices_snare = [4,9,14,19]
indices_hihat = [1,3,6,8,11,13,16,18]

for i in range(len(list_wav)):
    
    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets = onsets_samples.astype(int)

    audio = np.concatenate((audio,np.zeros(2048)))

    classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)

    if i<=9:
        sf.write('data/external/LVT_Audio/Dataset_Train_0' + str(i) + '.wav', audio, 44100)
        np.save('data/external/LVT_Audio/Onsets_Train_0' + str(i), onsets)
        np.save('data/external/LVT_Audio/Classes_Train_0' + str(i), classes)
    else:
        sf.write('data/external/LVT_Audio/Dataset_Train_' + str(i) + '.wav', audio, 44100)
        np.save('data/external/LVT_Audio/Onsets_Train_' + str(i), onsets)
        np.save('data/external/LVT_Audio/Classes_Train_' + str(i), classes)






# Create Test Dataset Audio

fs = 44100

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

indices_kick = [0,2,5,7,10,12,15,17]
indices_snare = [4,9,14,19]
indices_hihat = [1,3,6,8,11,13,16,18]

for i in range(len(list_wav)):
    
    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets = onsets_samples.astype(int)

    audio = np.concatenate((audio,np.zeros(2048)))

    classes = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)

    if i<=9:
        sf.write('data/external/LVT_Audio/Dataset_Test_0' + str(i) + '.wav', audio, 44100)
        np.save('data/external/LVT_Audio/Onsets_Test_0' + str(i), onsets)
        np.save('data/external/LVT_Audio/Classes_Test_0' + str(i), classes)
    else:
        sf.write('data/external/LVT_Audio/Dataset_Test_' + str(i) + '.wav', audio, 44100)
        np.save('data/external/LVT_Audio/Onsets_Test_' + str(i), onsets)
        np.save('data/external/LVT_Audio/Classes_Test_' + str(i), classes)