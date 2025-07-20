import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FDMI import band_ranges, bandpass_filter, extract_freq_features
from sklearn.preprocessing import MinMaxScaler
import zipfile
import io
import os

def run_fdmi_app():
    st.title("EEG Frequency Domain Feature Extractor")
    st.markdown("""
**How segmentation and feature extraction works:**
- Each uploaded CSV file contains signals from 14 channels.
- For each channel, the signal is split into the number of segments you select above.
- Frequency-domain features are extracted from each segment of each channel.
- The results table shows features for every segment, channel, and file.
""")

    uploaded_zip = st.file_uploader("Upload ZIP file containing EEG CSVs (organized by class folders)", type=["zip"])
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
    n_segments = st.number_input("Number of Segments", min_value=1, value=5)
    all_freq_features = []
    error_files = []

    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            class_folders = set([os.path.dirname(f) for f in z.namelist() if f.lower().endswith('.csv')])
            total_files = sum([len([f for f in z.namelist() if f.startswith(class_folder + '/') and f.lower().endswith('.csv')]) for class_folder in class_folders])
            progress = st.progress(0, text="Processing files...")
            processed_files = 0
            for class_folder in class_folders:
                class_label = os.path.basename(class_folder)
                csv_files = [
                    f for f in z.namelist()
                    if f.startswith(class_folder + '/')
                    and f.lower().endswith('.csv')
                    and not f.startswith('__MACOSX/')
                    and not os.path.basename(f).startswith('._')
                ]
                for csv_name in csv_files:
                    processed_files += 1
                    progress.progress(processed_files / total_files, text=f"Processing {csv_name} ({processed_files}/{total_files})")
                    with z.open(csv_name) as f:
                        try:
                            df = pd.read_csv(f)
                        except Exception as e:
                            error_files.append(f"{csv_name}: {e}")
                            continue
                        ch_names = df.columns.tolist()
                        data = df.values.T
                        total_samples = data.shape[1]
                        seg_len = total_samples // n_segments
                        for seg_idx in range(n_segments):
                            start = seg_idx * seg_len
                            stop = start + seg_len
                            seg_data = data[:, start:stop]
                            for ch_idx, ch_name in enumerate(ch_names):
                                signal = seg_data[ch_idx]
                                if len(signal) <= 27:
                                    continue  # Skip segments that are too short
                                # Cache filtered signals and band powers for this segment/channel
                                band_results = {}
                                for band, (low, high) in band_ranges.items():
                                    try:
                                        filtered = bandpass_filter(signal, low, high, fs)
                                        bp, ent, cent, peak, meanf, medf = extract_freq_features(filtered, fs)
                                        band_results[band] = {
                                            "filtered": filtered,
                                            "band_power": bp,
                                            "spectral_entropy": ent,
                                            "centroid": cent,
                                            "peak_freq": peak,
                                            "mean_freq": meanf,
                                            "median_freq": medf
                                        }
                                    except Exception as e:
                                        error_files.append(f"{csv_name} segment {seg_idx+1} channel {ch_name} band {band}: {e}")
                                        continue
                                # Compute total power and ratios only once per segment/channel
                                total_power = np.sum([res["band_power"] for res in band_results.values()])
                                alpha_power = band_results.get('alpha', {}).get('band_power', 0)
                                beta_power = band_results.get('beta', {}).get('band_power', 0)
                                theta_power = band_results.get('theta', {}).get('band_power', 0)
                                alpha_beta_ratio = alpha_power / beta_power if beta_power != 0 else 0
                                theta_beta_ratio = theta_power / beta_power if beta_power != 0 else 0
                                for band, res in band_results.items():
                                    rel_power = res["band_power"] / total_power if total_power != 0 else 0
                                    all_freq_features.append({
                                        "file": csv_name,
                                        "class": class_label,
                                        "segment": seg_idx + 1,
                                        "channel": ch_name,
                                        "band": band,
                                        "band_power": res["band_power"],
                                        "relative_power": rel_power,
                                        "alpha_beta_ratio": alpha_beta_ratio,
                                        "theta_beta_ratio": theta_beta_ratio,
                                        "spectral_entropy": res["spectral_entropy"],
                                        "centroid": res["centroid"],
                                        "peak_freq": res["peak_freq"],
                                        "mean_freq": res["mean_freq"],
                                        "median_freq": res["median_freq"]
                                    })
            progress.progress(1.0, text="Processing complete!")
            if error_files:
                st.error(f"Errors occurred in the following files/segments:\n" + "\n".join(error_files))

    selected_band = st.selectbox("Select Frequency Band", list(band_ranges.keys()))
    features_df = pd.DataFrame([f for f in all_freq_features if f["band"] == selected_band])
    if not features_df.empty and 'class' in features_df.columns:
        class_dummies = pd.get_dummies(features_df['class'])
        # Convert boolean columns to int (0/1)
        class_dummies = class_dummies.astype(int)
        features_df = pd.concat([features_df, class_dummies], axis=1)
        st.write(f"Features for {selected_band} band (with one-hot class labels):", features_df)
        feature_cols = [col for col in features_df.columns if col not in ['file', 'class', 'segment', 'channel', 'band'] + list(class_dummies.columns)]
        scaler = MinMaxScaler()
        features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
        st.write("Normalized Features (with one-hot class labels):", features_df)

        if st.button("Compute MI Scores (One-vs-Rest)"):
            class_names = features_df['class'].unique()
            mi_results = []
            X = features_df[feature_cols].values
            for class_name in class_names:
                y_binary = (features_df['class'] == class_name).astype(int)
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(X, y_binary, discrete_features=False)
                for feat, score in zip(feature_cols, mi_scores):
                    mi_results.append({
                        "Class": class_name,
                        "Feature": feat,
                        "MI Score": score
                    })
            mi_df = pd.DataFrame(mi_results).sort_values(["Class", "MI Score"], ascending=[True, False])
            st.write("Mutual Information Scores (Feature vs Each Class, One-vs-Rest):", mi_df)
            for class_name in class_names:
                class_mi = mi_df[mi_df["Class"] == class_name]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(class_mi["Feature"], class_mi["MI Score"], color='mediumseagreen')
                ax.set_xlabel("Mutual Information Score")
                ax.set_title(f"MI Scores for {selected_band.upper()} Band vs {class_name}")
                ax.invert_yaxis()
                st.pyplot(fig)

        # Save button for current band
        csv = features_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download Features CSV for {selected_band} Band", data=csv, file_name=f"features_{selected_band}.csv", mime="text/csv")
    else:
        st.info("No features found for the selected band or file. Please check your data and selection.")

# Ensure the app runs when executed
if __name__ == "__main__":
    run_fdmi_app()
