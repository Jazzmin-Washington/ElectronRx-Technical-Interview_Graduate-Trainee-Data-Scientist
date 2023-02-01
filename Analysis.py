#%%
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal
import scipy.stats
import sklearn.cluster
import sklearn.preprocessing
import heartpy as hp
import matplotlib.pyplot as plt
import os



def peak_detection(
    segment: np.ndarray, 
    distance: int,  
    prominence: int, 
    use_clustering: bool):

    peaks, properties = scipy.signal.find_peaks(
                                    segment, 
                                    distance=distance, 
                                    prominence=prominence, 
                                    height=0, 
                                    width=0)

    # Attempt to determine correct peaks by distinguishing the R wave from P and T waves
    if len(peaks) >= 3 and use_clustering:
        k_means = sklearn.cluster.KMeans(n_clusters=3).fit(
                                                        np.column_stack(
                                                            (properties["widths"], properties["peak_heights"], properties["prominences"])
                                                        )
                                                    )

        # Use width centroids to determine correct wave (least width, most prominence)
        # If the two lowest values are too close (< 5), use prominence to distinguish them
        width_cen = k_means.cluster_centers_[:, 0]
        labels_sort_width = np.argsort(width_cen)
        if width_cen[labels_sort_width[1]] - width_cen[labels_sort_width[0]] < 5:
            # Label of maximum prominence for lowest two widths
            prom_cen = k_means.cluster_centers_[:, 2]
            wave_label = np.argsort(prom_cen[labels_sort_width[:2]])[1]
        else:
            wave_label = labels_sort_width[0]

        is_wave_peak = k_means.labels_ == wave_label

        wave_peaks = peaks[is_wave_peak]
        wave_props = {k: v[is_wave_peak] for k, v in properties.items()}
    else:
        wave_peaks = peaks
        wave_props = properties

    # @PeterKirk does this need to be > 3 or >= 3?
    # Also, should this potentially be done before clustering?
    if len(wave_peaks) > 3:
        # Approximate prominences at edges of window
        base_height = segment[wave_peaks] - wave_props["prominences"]
        wave_props["prominences"][0] = wave_props["peak_heights"][0] - base_height[1]
        wave_props["prominences"][-1] = wave_props["peak_heights"][-1] - base_height[-2]
    return wave_peaks, wave_props


def frequency_domain(x, sfreq: int = 5):
    
    if len(x) < 4:  # RapidHRV edit: Can't run with less than 4 IBIs
        return np.nan

    # Interpolate R-R interval
    time = np.cumsum(x)
    f = scipy.interpolate.interp1d(time, x, kind="cubic")
    new_time = np.arange(time[0], time[-1], 1000 / sfreq)  # sfreq = 5 Hz
    x = f(new_time)

    # Define window length
    nperseg = 256 * sfreq
    if nperseg > len(x):
        nperseg = len(x)

    # Compute Power Spectral Density
    freq, psd = scipy.signal.welch(x=x, fs=sfreq, nperseg=nperseg, nfft=nperseg)
    psd = psd / 1000000
    fbands = {"hf": ("High frequency", (0.15, 0.4), "r")}

    # Extract HRV parameters
    ########################
    stats = pd.DataFrame(columns=["Values", "Metric"])
    band = "hf"

    this_psd = psd[(freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]
    this_freq = freq[(freq >= fbands[band][1][0]) & (freq < fbands[band][1][1])]

    if (len(this_psd) == 0) | (len(this_psd) == 0):  # RapidHRV edit: if no power
        return np.nan

    # Peaks (Hz)
    peak = round(this_freq[np.argmax(this_psd)], 4)
    stats.loc[len(stats)+1, :] = [peak, band + "_peak"]

    # Power (ms**2)
    power = np.trapz(x=this_freq, y=this_psd) * 1000000
    stats.loc[len(stats) + 1, :] = [power, band + "_power"]

    hf = stats.Values[stats.Metric == "hf_power"].values[0]

    return hf


def outlier_detection(
    peaks: np.ndarray,
    peak_properties: dict,
    ibi: np.ndarray,
    sample_rate: int,
    window_width: int,
    bpm: float,
    rmssd: float):

 
    bpm_in_range = 30  < bpm < 140
    rmssd_in_range = 5 < rmssd < 262

    if not (bpm_in_range and rmssd_in_range):
        return True

    max_peak_distance = (peaks[-1] - peaks[0]) / sample_rate
    print(max_peak_distance)
    if max_peak_distance < (window_width * 0.5):
        return True

    def mad_outlier_detection(x: np.ndarray, threshold: float) -> np.ndarray:
        x = x - np.mean(x)
        mad = scipy.stats.median_abs_deviation(x) * threshold
        return (x > mad) | (x < -mad)

    prominence_outliers = mad_outlier_detection(
        peak_properties["prominences"],5)
    if np.any(prominence_outliers):
        return True

    height_outliers = mad_outlier_detection(
        peak_properties["peak_heights"], 5)
    if np.any(height_outliers):
        return True

    ibi_outliers = mad_outlier_detection(ibi, 5)
    if np.any(ibi_outliers):
        return True

    return False
def analyze(
    data:np.array,
    sample_rate:int,
    window_width: int = 10,
    window_overlap: int = 0,
    ecg_prt_clustering: bool = False,
    amplitude_threshold: int = 50,
    distance_threshold: int = 250,
    n_required_peaks: int = 3) -> pd.DataFrame:
    
    DATA_COLUMNS = ["BPM", "RMSSD", "SDNN", "SDSD","IBI", "pNN20", "pNN50", "HF"]
    DATAFRAME_COLUMNS = ["Time", *DATA_COLUMNS, "Outlier", "Window"]

    

    if n_required_peaks < 3:
        raise ValueError("Parameter 'n_required_peaks' must be greater than three.")

    # Peak detection settings
    if ecg_prt_clustering:
        distance = 1
        prominence = 5
    else:
        distance = int((distance_threshold / 1000) * sample_rate)
        prominence = amplitude_threshold

    # Windowing function
    results = []
    for sample_start in range(
        0, len(data), (window_width - window_overlap) * sample_rate
    ):
        timestamp = sample_start / sample_rate

        segment = data[sample_start : sample_start + (window_width * sample_rate)]
        normalized = sklearn.preprocessing.minmax_scale(segment, (0, 100))
        peaks, properties = peak_detection(normalized, distance, prominence, ecg_prt_clustering)
        window_data = (normalized, peaks, properties)

        ibi = np.diff(peaks) * 1000 / sample_rate
        sd = np.diff(ibi)
            

        if len(peaks) <= n_required_peaks:
            results.append([timestamp, *[np.nan] * len(DATA_COLUMNS), True, window_data])
        else:
            # Time-domain metrics
            bpm = ((len(peaks) - 1) / ((peaks[-1] - peaks[0]) / sample_rate)) * 60
            rmssd = np.sqrt(np.mean(np.square(sd)))
            sdnn = np.std(ibi)
            sdsd = np.std(sd)  # Standard deviation of successive differences
            p_nn20 = np.sum(sd > 20) / len(sd)  # Proportion of successive differences > 20ms
            p_nn50 = np.sum(sd > 50) / len(sd)  # Proportion of successive differences > 50ms

            # Frequency-domain metrics
            hf = frequency_domain(x=ibi, sfreq=sample_rate)

            is_outlier = outlier_detection(
                peaks,
                properties,
                ibi,
                sample_rate,
                window_width,
                bpm,
                rmssd,
            )

            results.append(
                [timestamp, bpm, rmssd, sdnn, sdsd, ibi, p_nn20, p_nn50, hf, is_outlier, window_data]
            )

    return pd.DataFrame(
        results,
        columns=DATAFRAME_COLUMNS,
    )

def analyse_all_channels(file, dataframe, sample_rate):
    filename = file.strip('csv')
    new_dataframe = pd.DataFrame()

    for columns in dataframe:
        if columns == 'time'or columns == 'sample_rate':
            pass
        else:
            data = dataframe[columns].values
            print(data[0:5])
            
            analysis = analyze(data = dataframe[columns], sample_rate=sample_rate)
            print(analysis)



if __name__ == "__main__":
    writer = pd.ExcelWriter('full_raw_ppg_data.xlsx', engine = "xlsxwriter")
    os.chdir('/home/jazzy/Documents/PPS-Project/Filtered_PPG')
    for file in os.listdir():
        if ".csv" in file and ".png" not in file:
            ppg_signals = pd.read_csv(file)
            filename = file.strip('.csv').strip('filtered_')
            sample_rate=100
            for column in ['Red', "Green", 'Blue']:
                analyse = analyze(ppg_signals[column], sample_rate=int(sample_rate))
                analyse.to_csv(f'/home/jazzy/Documents/PPS-Project/Analysis/HeartRate_{filename}_{column}.csv')




# %%
