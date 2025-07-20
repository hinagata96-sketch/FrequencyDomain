import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
import mne
from sklearn.preprocessing import MinMaxScaler

# === Frequency Bands ===
band_ranges = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}

# === Bandpass Filter ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# === Frequency-Domain Features ===
def extract_freq_features(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    psd_sum = np.sum(psd)
    psd_norm = psd / psd_sum if psd_sum != 0 else psd

    band_power = psd_sum
    spec_entropy = entropy(psd_norm)
    centroid = np.sum(freqs * psd) / psd_sum if psd_sum != 0 else 0
    peak_freq = freqs[np.argmax(psd)]
    mean_freq = centroid
    median_freq = freqs[np.where(np.cumsum(psd) >= psd_sum / 2)[0][0]]

    return band_power, spec_entropy, centroid, peak_freq, mean_freq, median_freq

# === EEG Processor ===
def process_file_freq_features(file_path, emotion, subject, trial, fs=128, n_segments=5):
    df = pd.read_csv(file_path)
    ch_names = df.columns.tolist()
    data = df.values.T

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # ICA Artifact Removal
    ica = mne.preprocessing.ICA(n_components=min(15, len(ch_names)), random_state=42, method='fastica')
    ica.fit(raw)
    sources = ica.get_sources(raw).get_data()
    stds = np.std(sources, axis=1)
    threshold = np.percentile(stds, 90)
    ica.exclude = [i for i, s in enumerate(stds) if s > threshold]
    raw_clean = ica.apply(raw.copy())

    total_samples = raw_clean.n_times
    seg_len = total_samples // n_segments
    features = []

    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        stop = start + seg_len
        seg_data = raw_clean.get_data()[:, start:stop]

        for ch_idx, ch_name in enumerate(ch_names):
            signal = seg_data[ch_idx]

            for band, (low, high) in band_ranges.items():
                filtered = bandpass_filter(signal, low, high, fs)
                bp, ent, cent, peak, meanf, medf = extract_freq_features(filtered, fs)

                features.append({
                    "emotion": emotion,
                    "subject": subject,
                    "trial": trial,
                    "segment": seg_idx + 1,
                    "channel": ch_name,
                    "band": band,
                    "band_power": bp,
                    "spectral_entropy": ent,
                    "centroid": cent,
                    "peak_freq": peak,
                    "mean_freq": meanf,
                    "median_freq": medf
                })

    return features

## Batch processing and Colab-specific code removed for Streamlit compatibility.
## Use the functions above in your Streamlit app for interactive file processing.
