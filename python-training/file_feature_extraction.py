"""
All code is adapted from: http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

melspectrogram: Compute a Mel-scaled power spectrogram.
mfcc: Mel-frequency cepstral coefficients.
chorma-stft: Compute a chromagram from a waveform or power spectrogram
spectral_contrast: Compute spectral contrast.
tonnetz: Computes the tonal centroid features (tonnetz).
"""

# Requires librosa lib. for audio analysis
import glob
import os
import librosa
import numpy as np

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            
            # Splits are made along '-' so all laughter files should be named XXX-0-XXX.wav 
            # and all non-laughter files should be named XXX-1-XXX.wav or vice versa.
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
            
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
