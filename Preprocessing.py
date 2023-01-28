#%%
import dataclasses
from typing import Literal, Optional
import numpy as np
import scipy.interpolate
from scipy.signal import savgol_filter, cspline1d, cspline1d_eval
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

def cubic_spine_interpolaton(data):
    time = np.linspace(0, len(data))
    filtered_eval = cspline1d_eval(
                cspline1d(data), time)

    filtered = cspline1d(data)
    
    return filtered
    


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
    
def visualize(data,results):
    plt.figure(figsize=(12,12))
    plt.subplot(211)
    plt.plot(Red1[0:int(5 * sample_rate)])
    plt.title('original signal')
    plt.subplot(212)
    plt.plot(results[0:int(5 * sample_rate)])
    plt.title('filtered signal')
    plt.show()

    plt.figure(figsize=(12,6))
    plt.subplot(211)
    plt.plot(Red1[0:int(sample_rate * 15)])
    plt.title('15 second segment of unfiltered signal')
    plt.show()
    plt.figure(figsize=(12,6))
    plt.subplot(212)
    plt.plot(results[0:int(sample_rate * 15)])
    plt.title('15 second segment of filtered signal')
    plt.show()

def preprocess(data:[np.array],
                sample_rate:[int],
                butter_lowcut:[float] = 4,
                butter_order:[int] = 3,
                poly_order:[int] = 4,
                window_length:[int] = 101):

    nans = np.isnan(data)
    if np.any(nans):
        raise RuntimeError(
            "Cannot preprocess data containing NaN values. "
            f"First NaN found at index {nans.nonzero()[0][0]}."
        )
    result = cubic_spine_interpolaton(data)

    result = butter_lowpass_filter(data = result,
                                    sample_rate = sample_rate,
                                    lowcut = butter_lowcut,
                                    order = butter_order)
    
    smoothdata = savgol_filter(result,polyorder=poly_order, window_length=window_length)
    filtered= Red1-smoothdata
    
    return filtered

if __name__ == "__main__":
    os.chdir('/home/jazzy/Documents/PPS-Project')
    ppg_db_1 = pd.read_excel('full_raw_ppg_data.xlsx', sheet_name = 'ppg_db_1')
    ppg_db_1['time2'] = ppg_db_1['time'] * 1000
    Red1 = ppg_db_1['Red'].values
    Green1 = ppg_db_1['Green'].values
    Blue1 = ppg_db_1['Blue'].values
    time = ppg_db_1['time'].values
    timer = ppg_db_1['time2'].values

    sample_rate = hp.get_samplerate_mstimer(timer)
    print(sample_rate)
    filtereddata = preprocess(data = Red1, 
                        sample_rate = sample_rate)

    visualize(data = Red1, results=filtereddata)

# %%
