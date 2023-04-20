#Set up the notebook environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.stats import pearsonr
from scipy import signal as sig
from sklearn.feature_selection import SelectKBest, f_regression

def correlation(prediction, target):
    """Caluclate the correlation coefficient between prediction and target"""
    corr = [pearsonr(prediction[:,i], target[:,i]).statistic for i in range(4)]
    return corr, np.mean(corr)

def correlation_dl(prediction, target):
    """Caluclate the correlation coefficient between prediction and target"""
    corr = [pearsonr(prediction[:,i], target[:,i])[0] for i in range(4)]
    return corr, np.mean(corr)

def pack_submission(prediction_1, prediction_2, prediction_3):
    prediction_1 = np.insert(np.repeat(prediction_1, 50, axis=0), -1, [prediction_1[-1]]*50 , axis=0)
    prediction_2 = np.insert(np.repeat(prediction_2, 50, axis=0), -1, [prediction_2[-1]]*50 , axis=0)
    prediction_3 = np.insert(np.repeat(prediction_3, 50, axis=0), -1, [prediction_3[-1]]*50 , axis=0)

    prediction_1 = np.insert(prediction_1, 3, 0, axis=1)
    prediction_2 = np.insert(prediction_2, 3, 0, axis=1)
    prediction_3 = np.insert(prediction_3, 3, 0, axis=1)
    
    scipy.io.savemat('leaderboard_prediction.mat', {'predicted_dg': [[prediction_1], [prediction_2], [prediction_3]]})
    
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

def NumWins(x, fs, winLen, winDisp):
    return int(1 + (x.shape[0] - winLen * fs) / (winDisp * fs))

def LineLength(x):
    return np.abs(np.diff(x, axis=0)).sum(axis=0)

def Area(x):
    return np.abs(x).sum(axis=0)

def Energy(x):
    return (x ** 2).sum(axis=0)

def ZeroCrossingMean(x):
    return ((x < x.mean(axis=0))[1:] & (x[:-1] > x.mean(axis=0)) | (x > x.mean(axis=0))[1:] & (x[:-1] < x.mean(axis=0))).sum(axis=0)

def numSpikes(x):
    #TODO: implement
    sig.find_peaks(x, height=0, distance=100)
    pass

def averageTimeDomain(x):
    #TODO: implement
    return np.mean(x, axis=0)

def bandpower(x, fs, fmin, fmax):
    fs = 1000
    # win = 4 * sf
    freqs, psd = sig.welch(x, fs, axis=0, nperseg=x.shape[0])
    
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

def spectral_entropy(x, fs=1000):
    # Calculate the power spectrum
    f, Pxx = sig.welch(x, fs=fs)
    # Normalize the power spectrum
    Pxx_norm = Pxx / Pxx.sum()
    # Calculate the spectral entropy
    se = -1 * (Pxx_norm * np.log2(Pxx_norm)).sum()
    return se

def hjorth_complexity(x):
    dx = np.diff(x)
    d2x = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_d2x = np.var(d2x)
    activity = var_x
    mobility = np.sqrt(var_d2x / var_dx)
    # Calculate Hjorth complexity
    complexity = mobility / activity
    return complexity
    
# Kurtosis = @(x) ((1/size(x,1))*sum((x - mean(x)).^4))./(((1/size(x,1))*sum((x - mean(x)).^2)).^2);
def Kurtosis(x):
    return ((1/x.shape[0])*np.sum((x - np.mean(x))**4))/(((1/x.shape[0])*np.sum((x - np.mean(x))**2))**2)

def Covariance(x):
    convar = np.cov(x, rowvar=False)
    feat = []
    for i in range(convar.shape[0]):
        feat += [convar[i, :i]]
    return np.concatenate(feat)

def get_features(filtered_window, fs=1000):
    """
        Write a function that calculates features for a given filtered window. 
        Feel free to use features you have seen before in this class, features that
        have been used in the literature, or design your own!

        Input: 
        filtered_window (window_samples x channels): the window of the filtered ecog signal 
        fs: sampling rate
        Output:
        features (channels x num_features): the features calculated on each channel for the window
    """
    feat_LL = LineLength(filtered_window)
    feat_Area = Area(filtered_window)
    feat_Energy = Energy(filtered_window)
    feat_ZCM = ZeroCrossingMean(filtered_window)
    feat_TimeAvg = averageTimeDomain(filtered_window)
    feat_SpectralEntropy = spectral_entropy(filtered_window)
    feat_Hijorth = hjorth_complexity(filtered_window)
    feat_kurtosis = Kurtosis(filtered_window)
    feat_covariance = Covariance(filtered_window)
    # feat_FreqAvg = averageFreqDomain(filtered_window)
    
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    
    # covar = Covariances().fit_transform(np.expand_dims(filtered_window.T, 0))
    # # covest = Covariances('oas')
    # # temp = np.expand_dims(filtered_window, axis=-1)
    # # covar = covest.fit_transform(temp)
    # ts = TangentSpace()
    # tsfeat = ts.fit_transform(covar)
    # # print(tsfeat.shape)

    # raise notImplementedError()
    return np.hstack([feat_LL, 
                      feat_Area, 
                    #   feat_covariance,
                      feat_Energy, 
                      feat_ZCM, 
                      feat_TimeAvg, 
                      feat_SpectralEntropy,
                      feat_Hijorth,
                      feat_kurtosis,
                      feat_covariance,
                      bandpower(filtered_window, 1000, 5, 15),
                      bandpower(filtered_window, 1000, 20, 25),
                      bandpower(filtered_window, 1000, 75, 115),
                      bandpower(filtered_window, 1000, 125, 160),
                      bandpower(filtered_window, 1000, 160, 175)])
    
def get_windowed_feats(raw_ecog, fs, window_length, window_overlap):
    """
        Write a function which processes data through the steps of filtering and
        feature calculation and returns features. Points will be awarded for completing
        each step appropriately (note that if one of the functions you call within this script
        returns a bad output, you won't be double penalized). Note that you will need
        to run the filter_data and get_features functions within this function. 

        Inputs:
        raw_eeg (samples x channels): the raw signal
        fs: the sampling rate (1000 for this dataset)
        window_length: the window's length
        window_overlap: the window's overlap
        Output: 
        all_feats (num_windows x (channels x features)): the features for each channel for each time window
            note that this is a 2D array. 
    """
    raw_ecog = filter_data(raw_ecog, fs)
    
    window_disp = window_length - window_overlap
    
    all_feats = np.vstack([get_features(raw_ecog[int(i * window_disp * fs):int(i * window_disp * fs + window_length * fs), :], fs) for i in range(NumWins(raw_ecog, fs, window_length, window_disp))])
    
    return all_feats

def create_R_matrix(features, N_wind):
    """ 
    Write a function to calculate the R matrix

    Input:
        features (samples (number of windows in the signal) x channels x features): 
        the features you calculated using get_windowed_feats
        N_wind: number of windows to use in the R matrix

    Output:
        R (samples x (N_wind*channels*features))
    """
    num_win = features.shape[0]
    num_channel_features = features.shape[1]
    
    # Append a copy of the first N-1 rows to the beginning of features
    features = np.vstack((features[:N_wind-1], features))
    
    R = np.zeros((num_win, N_wind * num_channel_features))
    
    for i in range(num_win):
        # Get the feature matrix for the current window
        # Resize the feature matrix and store in R
        R[i,:] = features[i:i+N_wind,:].reshape(-1)

    R = np.hstack((np.ones((R.shape[0], 1)), R))

    return R
    
def feature_selection(R_train, y_train, num_features):
    selectors = [SelectKBest(f_regression, k=num_features//4) for _ in range(4)]
    for i in range(4):
        selectors[i].fit(R_train, y_train[:, i])
    idxs = [selector.get_support(indices=True) for selector in selectors]
    indices = np.unique(np.concatenate(idxs))
    return indices