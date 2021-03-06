import IPython.display as ipd
import soundfile as sf
import IPython
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.io.wavfile
import librosa
from librosa.util import frame
import os
#import rubberband
import pyrubberband as pyrb

from utils import Create_Phoneme_Labels, pitch_shift, time_stretch



# Create AVP Test Dataset Audio

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

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_samples = onsets_samples.astype(int)
            
    audio = np.concatenate((audio,np.zeros(1024)))

    audios_all = []
    for osm in range(len(onsets_samples)-1):
        audios_all.append(audio[onsets_samples[osm]:onsets_samples[osm+1]])
    audios_all.append(audio[onsets_samples[osm+1]:])

    np.save('data/external/AVP_Dataset_Audio/Dataset_Test_' + str(i).zfill(2), np.array(audios_all))

    if i==0:
        for pl in range(10):
            plt.figure()
            plt.plot(np.array(audios_all[pl]))
            plt.show()



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

            audio = np.concatenate((audio,np.zeros(1024)))
    
            for osm in range(len(onsets)-1):
                audios_all.append(audio[onsets[osm]:onsets[osm+1]])
            audios_all.append(audio[onsets[osm+1]:])

    np.save('data/external/AVP_Dataset_Audio/Dataset_Train_' + str(part).zfill(2), np.array(audios_all))







# Create LVT Test Dataset Audio

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

for i in range(len(list_wav)):

    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_samples = onsets_samples.astype(int)
            
    audio = np.concatenate((audio,np.zeros(1024)))

    audios_all = []
    for osm in range(len(onsets_samples)-1):
        audios_all.append(audio[onsets_samples[osm]:onsets_samples[osm+1]])
    audios_all.append(audio[onsets_samples[osm+1]:])

    np.save('data/external/LVT_Dataset_Audio/Dataset_Test_' + str(i).zfill(2), np.array(audios_all))

    if i==0:
        for pl in range(10):
            plt.figure()
            plt.plot(np.array(audios_all[pl]))
            plt.show()



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

for i in range(len(list_wav)):

    onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

    audio, fs = librosa.load(list_wav[i], sr=44100)
    audio_ref = audio/np.max(abs(audio))

    onsets_samples = onsets*fs
    onsets_ref = onsets_samples.astype(int)

    audios_all = []
    
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

        audio = np.concatenate((audio,np.zeros(1024)))

        for osm in range(len(onsets)-1):
            audios_all.append(audio[onsets[osm]:onsets[osm+1]])
        audios_all.append(audio[onsets[osm+1]:])

    np.save('data/external/LVT_Dataset_Audio/Dataset_Train_' + str(i).zfill(2), np.array(audios_all))