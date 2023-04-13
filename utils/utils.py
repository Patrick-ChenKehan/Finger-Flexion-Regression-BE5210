#Set up the notebook environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.stats import pearsonr
from scipy import signal as sig

def correlation(prediction, target):
    """Caluclate the correlation coefficient between prediction and target"""
    corr = [pearsonr(prediction[:,i], target[:,i]).statistic for i in range(4)]
    return corr, np.mean(corr)

def filter_data(raw_eeg, fs=1000):
    """
    Write a filter function to clean underlying data.
    Filter type and parameters are up to you. Points will be awarded for reasonable filter type, parameters and application.
    Please note there are many acceptable answers, but make sure you aren't throwing out crucial data or adversly
    distorting the underlying data!

    Input: 
        raw_eeg (samples x channels): the raw signal
        fs: the sampling rate (1000 for this dataset)
    Output: 
        clean_data (samples x channels): the filtered signal
    """
    dim = 100
    # b = sig.firwin(numtaps=dim + 1, cutoff=[0.15, 200], pass_zero='bandpass', fs=fs)
    b, a = sig.butter(N=2, Wn=[0.15, 200], btype='bandpass', fs=fs, output='ba')
    filtered_eeg = sig.filtfilt(b, a, x=raw_eeg, axis=0)
    
    return filtered_eeg

def concatenate(feats):
    new_features = np.zeros((feats.shape[0], feats.shape[1] * 2))

    for i in range(0, feats.shape[0]):
        if i > 0:
            new_features[i, 0: feats.shape[1] // 2 + 1] = feats[i - 1, -feats.shape[1] // 2:]
        new_features[i, feats.shape[1] // 2 + 1: 3 * feats.shape[1] // 2 + 1] = feats[i]
        if i < feats.shape[0] - 1:
            new_features[i, 3 * feats.shape[1] // 2 + 1:] = feats[i + 1, :feats.shape[1] // 2]
    return new_features

def LineLength(x):
    return np.abs(np.diff(x, axis=0)).sum(axis=-2)

def Area(x):
    return np.abs(x).sum(axis=-2)

def Energy(x):
    return (x ** 2).sum(axis=-2)

def ZeroCrossingMean(x):
    return ((x < x.mean(axis=0))[1:] & (x[:-1] > x.mean(axis=0)) | (x > x.mean(axis=0))[1:] & (x[:-1] < x.mean(axis=0))).sum(axis=-2)

def numSpikes(x):
    #TODO: implement
    sig.find_peaks(x, height=0, distance=100)
    pass

def averageTimeDomain(x):
    #TODO: implement
    return np.mean(x, axis=-2)

def bandpower(x, fs, fmin, fmax):
    fs = 1000
    # win = 4 * sf
    freqs, psd = sig.welch(x, fs, axis=-2, nperseg=x.shape[0])
    
    # Define delta lower and upper limits
    # fmin, fmax = 0.5, 4

    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    from scipy.integrate import simps

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

    # Compute the absolute power by approximating the area under the curve
    delta_power = simps(psd[idx_delta], dx=freq_res, axis=0)
    
    return delta_power