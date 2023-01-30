#%%
import dataclasses
from typing import Literal, Optional
import numpy as np
import scipy.interpolate
from scipy.signal import savgol_filter,  morlet, cspline1d, filtfilt, cspline1d_eval,detrend, butter, cheby2, sosfilt
from numpy import repeat
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
import os



def cubic_spine_interpolaton(data):
    time = np.linspace(0, len(data))
    filtered_eval = cspline1d_eval(
                cspline1d(data), time)

    filtered = cspline1d(data)
    
    return filtered

from scipy import signal

def moving_average_filter(x, n = 5):
    b = repeat(1.0/n, n) #Create impulse response
    xf = signal.lfilter(b, 1, x) #Filter the signal
    return(xf)


def butter_lowpass_filter(data, 
                        sample_rate, 
                        lowcut = 4, 
                        order = 3):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    #high = highcut / nyq
    b, a = scipy.signal.butter(order, [low], btype='low')
    y = scipy.signal.filtfilt(b, a, data)
    
    return y

def cheby2_filter(data, 
                sample_rate, 
                band = [3.5,7.8], 
                order:int = 4):

    sos = cheby2(order,20, band, 'band', 
                        fs = sample_rate,output='sos',
                        analog=False)
    filtered = filtfilt(sos, data)
    return filtered
    
def visualize(data,results, sample_rate):
    plt.figure(figsize=(12,12))
    plt.subplot(211)
    plt.plot(data[0:int(15 * sample_rate)])
    plt.title('original signal')
    plt.subplot(212)
    plt.plot(results[0:int(15 * sample_rate)])
    plt.title('filtered signal')
    plt.show()

    plt.figure(figsize=(12,6))
    plt.subplot(211)
    plt.plot(data[0:len(data)])
    plt.title('15 second segment of unfiltered signal')
    plt.show()
    plt.figure(figsize=(12,6))
    plt.subplot(212)
    plt.plot(results[0:len(results)])
    plt.title('15 second segment of filtered signal')
    plt.show()

def preprocess(data:[np.array],
                sample_rate:[int],
                moving_average = True,
                butter = True,
                butter_lowcut:[int] = 4,
                butter_order:[int] = 3,
                trendline = True,
                trend_cut = 0.05,
                sd_poly_order:[int] = 4,
                sg_window_length:[int] = 101,
                cheby2 = False,
                cheby_order = 4,
                cheby_band = [0.5, 6]):

    nans = np.isnan(data)
    if np.any(nans):
        raise RuntimeError(
            "Cannot preprocess data containing NaN values. "
            f"First NaN found at index {nans.nonzero()[0][0]}."
        )
    
    
    #result = moving_average_filter(data)
    result = detrend(data)
    if trendline == True:
        result = hp.remove_baseline_wander(result, sample_rate, cutoff = trend_cut)
    result = cubic_spine_interpolaton(result)

    #result = morlet(len(result), w = 6, complete=True)

    if butter == True:    
        result = butter_lowpass_filter(data = result,
                                    sample_rate = sample_rate,
                                    lowcut = butter_lowcut)
    if cheby2 == True:
        result = cheby2_filter(result, sample_rate=sample_rate, order=4)

                           
    
    #smoothdata = savgol_filter(result,polyorder=sd_poly_order, 
                                    #window_length= sg_window_length)
    #result = data-smoothdata
    return result
    
def preprocess_all_channels(file, dataframe, visualise:bool = False):
    filename = file
    timer = dataframe['time'].values * 1000
    sample_rate = hp.get_samplerate_mstimer(timer)
    print(sample_rate)
    filtered_ppg = pd.DataFrame()
    filtered_ppg['time'] = dataframe['time']
    filtered_ppg['sample_rate'] = sample_rate
    

    for columns in dataframe:
        if columns == 'time':
            pass
        
        else:
            data = dataframe[columns].values
            filtereddata = preprocess(data = data, 
                        sample_rate = sample_rate)
            filtered_ppg[columns] = filtereddata
        

            
    
    filtered_ppg.to_csv(f'/home/jazzy/Documents/PPS-Project/Filtered_PPG/filtered_{filename}')

    if visualise == True:

        Red = dataframe['Red']
        Green = dataframe['Green']
        Blue = dataframe['Blue']

        

        Filtered_Red =  filtered_ppg['Red']
        Filtered_Green = filtered_ppg['Green']
        Filtered_Blue = filtered_ppg['Blue']

        Filtered_Green = hp.flip_signal(Filtered_Green)
        Filtered_Red = hp.flip_signal(Filtered_Red)
        #Filtered_Blue = hp.flip_signal(Filtered_Blue)

        fig,axs = plt.subplots(3,2)
        axs[0, 0].plot(Red[1500:1500+int(5 * sample_rate)], c = '#d55e00', label = 'Red Channel')
        axs[0, 0].set_title('Unfiltered Signals')
        axs[0, 1].plot(Filtered_Red[1500:1500+int(5 * sample_rate)],c ='#d55e00', label = 'Red Channel')
        axs[0, 1].set_title('Filtered Red Signal')
        axs[1, 0].plot(Green[1500:1500+int(5* sample_rate)], c ='#009e73', label = 'Green Channel')
        axs[1, 1].plot(Filtered_Green[1500:1500+int(5 * sample_rate)],c ='#009e73', label = 'Green Channel')
        axs[2, 0].plot(Blue[1500:1500+int(5 * sample_rate)],c = '#0072b2', label= 'Blue Channel')
        axs[2, 1].plot(Filtered_Blue[1500:1500+int(5 * sample_rate)],c = '#0072b2' , label = 'Blue Channel')
        for ax in axs.flat:
            ax.set(xlabel='PPG Signal', ylabel='Frequency (Hz)')
            ax.label_outer()

        
        plt.savefig(f'/home/jazzy/Documents/PPS-Project/Filtered_PPG/cb_{filename}.png', dpi='figure', format = 'png')
        plt.show()
        filename = filename.strip('.csv')
        
        

if __name__ == "__main__":
    os.chdir('/home/jazzy/Documents/PPS-Project/Raw_PPG_Files')
    file = '2022-06-14_12-55-43.csv'
    dataframe = pd.read_csv(file, index_col=0)
    preprocess_all_channels(file, dataframe, visualise=True)
# %%
