import os
import pdb
import numpy as np

import essentia
from essentia.standard import *


class extract_features():
    
    def __init__(self):
        
        self.num_extra_features = 17
        self.num_mel_coeffs = 13
        self.num_mel_bands = 40
        self.num_moments = 4
        self.num_features = 4+(self.num_mel_bands)+(self.num_mel_coeffs*6)+(self.num_extra_features*8)
        
    def compute(self, audio, framesize):
        
        hopsize = 256
        audio = essentia.array(audio)
        
        # Define preprocessing functions
        
        win = Windowing(type='hann')
        spec = Spectrum()
        
        # Define extractors
        
        mfcc = MFCC(highFrequencyBound=22050,numberCoefficients=self.num_mel_coeffs,numberBands=self.num_mel_bands)
        
        rolloff_25 = RollOff(cutoff=0.25)
        rolloff_50 = RollOff(cutoff=0.50)
        rolloff_90 = RollOff(cutoff=0.90)
        rolloff_95 = RollOff(cutoff=0.95)
        
        scx = SpectralComplexity()
        hfc = HFC()
        strp = StrongPeak()
        moments = CentralMoments()
        crest = Crest()
        decrease = Decrease()
        entropy = Entropy()
        flatness = Flatness()
        rms = RMS()
        zcr = ZeroCrossingRate()
        
        env = Envelope(applyRectification=True, attackTime=5, releaseTime=100)
        der = DerivativeSFX()
        flat = FlatnessSFX()
        tct = TCToTotal()
        
        # Extract features
        
        feature_vector = np.zeros(self.num_features)
        
        _mfccs = []
        _bands = []
        _rolloff_25 = []
        _rolloff_50 = []
        _rolloff_90 = []
        _rolloff_95 = []
        _scx = []
        _hfc = []
        _strp = []
        _moments = []
        _crest = []
        _decrease = []
        _entropy = []
        _flatness = []
        _rms = []
        _zcr = []
        
        envelope = env(audio)
        feature_vector[0], feature_vector[1] = der(envelope)
        feature_vector[2] = flat(envelope)
        feature_vector[3] = tct(envelope)
        
        for frame in FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
            _bandsv, _mfccv = mfcc(spec(win(frame)))
            _mfccs.append(_mfccv)
            _bands.append(_bandsv)
            _rolloff_25v = rolloff_25(spec(win(frame)))
            _rolloff_25.append(_rolloff_25v)
            _rolloff_50v = rolloff_50(spec(win(frame)))
            _rolloff_50.append(_rolloff_50v)
            _rolloff_90v = rolloff_90(spec(win(frame)))
            _rolloff_90.append(_rolloff_90v)
            _rolloff_95v = rolloff_95(spec(win(frame)))
            _rolloff_95.append(_rolloff_95v)
            _scxv = scx(spec(win(frame)))
            _scx.append(_scxv)
            _hfcv = hfc(spec(win(frame)))
            _hfc.append(_hfcv)
            _strpv = strp(spec(win(frame)))
            _strp.append(_strpv)
            _momentsv = moments(spec(win(frame)))
            _moments.append(_momentsv)
            _crestv = crest(spec(win(frame)))
            _crest.append(_crestv)
            _decreasev = decrease(spec(win(frame)))
            _decrease.append(_decreasev)
            _entropyv = entropy(spec(win(frame)))
            _entropy.append(_entropyv)
            _flatnessv = flatness(spec(win(frame)))
            _flatness.append(_flatnessv)
            _rmsv = rms(spec(win(frame)))
            _rms.append(_rmsv)
            _zcrv = zcr(frame)
            _zcr.append(_zcrv)
            
        # Allocate features

        I = 4 
            
        for i in range(self.num_mel_coeffs):
            feature_vector[I+i] = np.mean(np.array(_mfccs)[:,i])
            feature_vector[I+self.num_mel_coeffs+i] = np.std(np.array(_mfccs)[:,i])
            feature_vector[I+self.num_mel_coeffs*2+i] = np.mean(np.gradient(np.array(_mfccs)[:,i]))
            feature_vector[I+self.num_mel_coeffs*3+i] = np.var(np.gradient(np.array(_mfccs)[:,i]))
            feature_vector[I+self.num_mel_coeffs*4+i] = np.mean(np.gradient(np.gradient(np.array(_mfccs)[:,i])))
            feature_vector[I+self.num_mel_coeffs*5+i] = np.var(np.gradient(np.gradient(np.array(_mfccs)[:,i])))
            
        I = 4 + self.num_mel_coeffs*6
            
        for i in range(self.num_mel_bands):
            feature_vector[I+i] = np.mean(np.array(_bands)[:,i])
            
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands
        
        feature_vector[I] = np.mean(np.array(_rolloff_25))
        feature_vector[I+1] = np.mean(np.array(_rolloff_50))
        feature_vector[I+2] = np.mean(np.array(_rolloff_90))
        feature_vector[I+3] = np.mean(np.array(_rolloff_95))
        feature_vector[I+4] = np.mean(np.array(_scx))
        feature_vector[I+5] = np.mean(np.array(_hfc))
        feature_vector[I+6] = np.mean(np.array(_strp))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.mean(np.array(moment_array))
        feature_vector[I+11] = np.mean(np.array(_crest))
        feature_vector[I+12] = np.mean(np.array(_decrease))
        feature_vector[I+13] = np.mean(np.array(_entropy))
        feature_vector[I+14] = np.mean(np.array(_flatness))
        feature_vector[I+15] = np.mean(np.array(_rms))
        feature_vector[I+16] = np.mean(np.array(_zcr))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features
        
        feature_vector[I] = np.var(np.array(_rolloff_25))
        feature_vector[I+1] = np.var(np.array(_rolloff_50))
        feature_vector[I+2] = np.var(np.array(_rolloff_90))
        feature_vector[I+3] = np.var(np.array(_rolloff_95))
        feature_vector[I+4] = np.var(np.array(_scx))
        feature_vector[I+5] = np.var(np.array(_hfc))
        feature_vector[I+6] = np.var(np.array(_strp))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.var(np.array(moment_array))
        feature_vector[I+11] = np.var(np.array(_crest))
        feature_vector[I+12] = np.var(np.array(_decrease))
        feature_vector[I+13] = np.var(np.array(_entropy))
        feature_vector[I+14] = np.var(np.array(_flatness))
        feature_vector[I+15] = np.var(np.array(_rms))
        feature_vector[I+16] = np.var(np.array(_zcr))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*2
        
        feature_vector[I] = np.min(np.array(_rolloff_25))
        feature_vector[I+1] = np.min(np.array(_rolloff_50))
        feature_vector[I+2] = np.min(np.array(_rolloff_90))
        feature_vector[I+3] = np.min(np.array(_rolloff_95))
        feature_vector[I+4] = np.min(np.array(_scx))
        feature_vector[I+5] = np.min(np.array(_hfc))
        feature_vector[I+6] = np.min(np.array(_strp))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.min(np.array(moment_array))
        feature_vector[I+11] = np.min(np.array(_crest))
        feature_vector[I+12] = np.min(np.array(_decrease))
        feature_vector[I+13] = np.min(np.array(_entropy))
        feature_vector[I+14] = np.min(np.array(_flatness))
        feature_vector[I+15] = np.min(np.array(_rms))
        feature_vector[I+16] = np.min(np.array(_zcr))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*3
        
        feature_vector[I] = np.max(np.array(_rolloff_25))
        feature_vector[I+1] = np.max(np.array(_rolloff_50))
        feature_vector[I+2] = np.max(np.array(_rolloff_90))
        feature_vector[I+3] = np.max(np.array(_rolloff_95))
        feature_vector[I+4] = np.max(np.array(_scx))
        feature_vector[I+5] = np.max(np.array(_hfc))
        feature_vector[I+6] = np.max(np.array(_strp))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.max(np.array(moment_array))
        feature_vector[I+11] = np.max(np.array(_crest))
        feature_vector[I+12] = np.max(np.array(_decrease))
        feature_vector[I+13] = np.max(np.array(_entropy))
        feature_vector[I+14] = np.max(np.array(_flatness))
        feature_vector[I+15] = np.max(np.array(_rms))
        feature_vector[I+16] = np.max(np.array(_zcr))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*4
        
        feature_vector[I] = np.mean(np.gradient(np.array(_rolloff_25)))
        feature_vector[I+1] = np.mean(np.gradient(np.array(_rolloff_50)))
        feature_vector[I+2] = np.mean(np.gradient(np.array(_rolloff_90)))
        feature_vector[I+3] = np.mean(np.gradient(np.array(_rolloff_95)))
        feature_vector[I+4] = np.mean(np.gradient(np.array(_scx)))
        feature_vector[I+5] = np.mean(np.gradient(np.array(_hfc)))
        feature_vector[I+6] = np.mean(np.gradient(np.array(_strp)))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.mean(np.gradient(np.array(moment_array)))
        feature_vector[I+11] = np.mean(np.gradient(np.array(_crest)))
        feature_vector[I+12] = np.mean(np.gradient(np.array(_decrease)))
        feature_vector[I+13] = np.mean(np.gradient(np.array(_entropy)))
        feature_vector[I+14] = np.mean(np.gradient(np.array(_flatness)))
        feature_vector[I+15] = np.mean(np.gradient(np.array(_rms)))
        feature_vector[I+16] = np.mean(np.gradient(np.array(_zcr)))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*5
        
        feature_vector[I] = np.var(np.gradient(np.array(_rolloff_25)))
        feature_vector[I+1] = np.var(np.gradient(np.array(_rolloff_50)))
        feature_vector[I+2] = np.var(np.gradient(np.array(_rolloff_90)))
        feature_vector[I+3] = np.var(np.gradient(np.array(_rolloff_95)))
        feature_vector[I+4] = np.var(np.gradient(np.array(_scx)))
        feature_vector[I+5] = np.var(np.gradient(np.array(_hfc)))
        feature_vector[I+6] = np.var(np.gradient(np.array(_strp)))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.var(np.gradient(np.array(moment_array)))
        feature_vector[I+11] = np.var(np.gradient(np.array(_crest)))
        feature_vector[I+12] = np.var(np.gradient(np.array(_decrease)))
        feature_vector[I+13] = np.var(np.gradient(np.array(_entropy)))
        feature_vector[I+14] = np.var(np.gradient(np.array(_flatness)))
        feature_vector[I+15] = np.var(np.gradient(np.array(_rms)))
        feature_vector[I+16] = np.var(np.gradient(np.array(_zcr)))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*6
        
        feature_vector[I] = np.min(np.gradient(np.array(_rolloff_25)))
        feature_vector[I+1] = np.min(np.gradient(np.array(_rolloff_50)))
        feature_vector[I+2] = np.min(np.gradient(np.array(_rolloff_90)))
        feature_vector[I+3] = np.min(np.gradient(np.array(_rolloff_95)))
        feature_vector[I+4] = np.min(np.gradient(np.array(_scx)))
        feature_vector[I+5] = np.min(np.gradient(np.array(_hfc)))
        feature_vector[I+6] = np.min(np.gradient(np.array(_strp)))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.min(np.gradient(np.array(moment_array)))
        feature_vector[I+11] = np.min(np.gradient(np.array(_crest)))
        feature_vector[I+12] = np.min(np.gradient(np.array(_decrease)))
        feature_vector[I+13] = np.min(np.gradient(np.array(_entropy)))
        feature_vector[I+14] = np.min(np.gradient(np.array(_flatness)))
        feature_vector[I+15] = np.min(np.gradient(np.array(_rms)))
        feature_vector[I+16] = np.min(np.gradient(np.array(_zcr)))
        
        I = 4 + self.num_mel_coeffs*6 + self.num_mel_bands + self.num_extra_features*7
        
        feature_vector[I] = np.max(np.gradient(np.array(_rolloff_25)))
        feature_vector[I+1] = np.max(np.gradient(np.array(_rolloff_50)))
        feature_vector[I+2] = np.max(np.gradient(np.array(_rolloff_90)))
        feature_vector[I+3] = np.max(np.gradient(np.array(_rolloff_95)))
        feature_vector[I+4] = np.max(np.gradient(np.array(_scx)))
        feature_vector[I+5] = np.max(np.gradient(np.array(_hfc)))
        feature_vector[I+6] = np.max(np.gradient(np.array(_strp)))
        for i in range(self.num_moments):
            moment_array = []
            for j in range(len(_moments)):
                moment_array.append(_moments[j][1+i])
            feature_vector[I+7+i] = np.max(np.gradient(np.array(moment_array)))
        feature_vector[I+11] = np.max(np.gradient(np.array(_crest)))
        feature_vector[I+12] = np.max(np.gradient(np.array(_decrease)))
        feature_vector[I+13] = np.max(np.gradient(np.array(_entropy)))
        feature_vector[I+14] = np.max(np.gradient(np.array(_flatness)))
        feature_vector[I+15] = np.max(np.gradient(np.array(_rms)))
        feature_vector[I+16] = np.max(np.gradient(np.array(_zcr)))
        
        return feature_vector
    
    
    
frame_sizes = [512,1024,2048]

for frame_size in frame_sizes:

    for part in range(28):

        if part<=9:
            list_audio = np.load('../Data/UC_AVP_Audio/Dataset_Test_0' + str(part) + '.npy', allow_pickle=True)
        else:
            list_audio = np.load('../Data/UC_AVP_Audio/Dataset_Test_' + str(part) + '.npy', allow_pickle=True)

        extractor = extract_features()

        for frame_size in frame_sizes:

            features = np.zeros((list_audio.shape[0],extractor.num_features))

            for n in range(list_audio.shape[0]):

                features[n] = extractor.compute(list_audio[n],frame_size)

        np.save('features/Features_Engineered_Test_' + str(frame_size) + '_' + str(part), features)

    for part in range(28):

        if part<=9:
            list_audio = np.load('../Data/UC_AVP_Audio/Dataset_Train_0' + str(part) + '.npy', allow_pickle=True)
        else:
            list_audio = np.load('../Data/UC_AVP_Audio/Dataset_Train_' + str(part) + '.npy', allow_pickle=True)

        extractor = extract_features()

        for frame_size in frame_sizes:

            features = np.zeros((list_audio.shape[0],extractor.num_features))

            for n in range(list_audio.shape[0]):

                features[n] = extractor.compute(list_audio[n],frame_size)

        np.save('features/Features_Engineered_Train_Aug_' + str(frame_size) + '_' + str(part), features)


