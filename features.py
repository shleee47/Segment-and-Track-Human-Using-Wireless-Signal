import pdb
import torch
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


class Feature_Extractor():
    def __init__(self, n_fft=512, hopsize=128, window='hann'):
        self.nfft = n_fft
        self.hopsize = 128
        self.window = 'hann'
        self.melW = 128
        self.n_mfcc= 40

    def stft(self,sig):
        #pdb.set_trace()
        S = np.abs(librosa.stft(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect'))**2  
        return S

    def mel(self,sig):
        #pdb.set_trace()
        S = librosa.feature.melspectrogram(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
        return S

    def logmel(self,sig):
        #pdb.set_trace()
        S = librosa.feature.melspectrogram(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
        S = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None)
        return S

    def MFCC(self, sig): 
        S = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=self.n_mfcc,
                n_fft = self.nfft, hop_length = self.hopsize)
        return S

        return S

    def Poly(self, sig, order):
        S = librosa.feature.poly_features(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize,
                                        order=order)
        return S

    def Chroma(self, sig):
        S = librosa.feature.chroma_stft(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def spectral_centroid(self, sig):
        S = librosa.feature.spectral_centroid(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def spectral_bandwidth(self, sig):
        S = librosa.feature.spectral_bandwidth(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S
        
    def spectral_contrast(self, sig):
        S = librosa.feature.spectral_contrast(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def spectral_flatness(self, sig):
        S = librosa.feature.spectral_flatness(y=sig,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def spectral_rolloff(self, sig):
        S = librosa.feature.spectral_rolloff(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def tonnetz(self, sig):
        S = librosa.feature.tonnetz(y=sig,
                                    hop_length = self.hopsize,
                                    sr=16000)
        return S

    def Chroma_cqt(self, sig):
        S = librosa.feature.chroma_cqt(y=sig,
                                        sr=16000,
                                        hop_length=self.hopsize)
        return S

    def Chroma_cens(self, sig):
        S = librosa.feature.chroma_cens(y=sig,
                                        sr=16000,
                                        hop_length=self.hopsize)
        return S