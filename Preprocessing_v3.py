#%%
#

import neurokit2 as nk
import pandas as pd
import numpy as np
import heartpy as hp
import pickle
import csv
import sklearn
from sklearn.metrics import pairwise_distances
import os

os.chdir('/home/jazzy/Documents/PPS-Project/Raw_PPG_Files')
file = '2022-06-14_12-55-43.csv'
ppg_signal = pd.read_csv(file, index_col = 0)
timer = ppg_signal['time'].values * 1000
sampling_rate = hp.get_samplerate_mstimer(timer)



ppg_data = nk.ppg_clean(ppg_signal['Red'], sampling_rate)
df, info = nk.ppg_process(ppg_data, sampling_rate, report ='my_report.html')
time = nk.hrv_time(df,sampling_rate, show = True)
freq = nk.hrv_frequency(df, sampling_rate, show = True)
shannon = nk.entropy_shannon(ppg_data, symbolize=3, show=True)
entropy_svd = nk.entropy_svd(ppg_data)
entropy_sample = nk.entropy_sample(ppg_data)
final_analysis = pd.concat([df,time])
final_analysis = pd.concat([final_analysis, freq])
out = {'shannon':shannon, 'SampEn':entropy_sample}


f, t, cwtm = nk.signal_timefrequency(ppg_data, int(sampling_rate), method = 'cwt', max_frequency = 10,
    mode='complex',overlap= 5, show = True)
time_freq_analysis = dict({'freq':f, 'time':t, "cwtm":cwtm})


os.chdir('/home/jazzy/Documents/PPS-Project/Analysis')
with open(f'time_freq_wavelet_{file}', 'w', newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames = time_freq_analysis.keys)
out = pd.DataFrame(out)
final_analysis = pd.concat([final_analysis,out])
final_analysis.to_csv(f'final_analysis_{file}') 
ppg_analysis = df.to_csv(f'ppg_analysis_{file}')



        
# %%
