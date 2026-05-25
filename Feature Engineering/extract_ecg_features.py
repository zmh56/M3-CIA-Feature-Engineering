import argparse
import os
import glob
import pandas as pd
import numpy as np
import scipy.io
import neurokit2 as nk
import pywt
from scipy.signal import butter, filtfilt, detrend, medfilt, hilbert
from scipy.stats import skew, kurtosis

# ==========================================
# 1. Preprocessing Functions
# ==========================================

def remove_baseline_wander(ecg_signal, window_size):
    """
    Remove baseline wander using median filter and detrending.
    """
    window_size = window_size + 1 if (window_size % 2 == 0) else window_size
    ecg_baseline = medfilt(ecg_signal, window_size)
    detrended_signal = detrend(ecg_signal - ecg_baseline)
    return detrended_signal

def lowpass_filter(ecg_signal, cutoff_freq=25, fs=250, order=5):
    """
    Apply zero-phase Butterworth lowpass filter.
    """
    nyquist = fs / 2.0
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, ecg_signal)

def highpass_filter(ecg_signal, cutoff_freq=0.5, fs=250, order=3):
    """
    Apply zero-phase Butterworth highpass filter.
    """
    nyquist = fs / 2.0
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='high')
    return filtfilt(b, a, ecg_signal)

def wavelet_denoise(ecg_signal, wavelet_name='db6', level=5):
    """
    Denoise ECG signal using discrete wavelet transform (soft thresholding).
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet_name, level=level)
    threshold = (
        np.median(np.abs(coeffs[-1])) / 0.6745
        * np.sqrt(2 * np.log(len(ecg_signal)))
        * 0.85
    )
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet_name)[:len(ecg_signal)]

def preprocess_ecg(ecg_signal, sampling_rate=250):
    """
    Standard preprocessing pipeline for ECG signals.
    """
    window_size = int(2.0 * sampling_rate)
    
    # 1. Baseline wander removal
    ecg_bw_removed = remove_baseline_wander(ecg_signal, window_size)
    
    # 2. Lowpass filtering (remove high-frequency noise/muscle artifacts)
    ecg_lp = lowpass_filter(ecg_bw_removed, cutoff_freq=25, fs=sampling_rate)
    
    # 3. Wavelet denoising
    ecg_denoised = wavelet_denoise(ecg_lp)
    
    # 4. Highpass filtering (remove remaining low-frequency trends)
    ecg_clean = highpass_filter(ecg_denoised, cutoff_freq=0.5, fs=sampling_rate)
    
    return ecg_clean

# ==========================================
# 2. Feature Extraction Functions
# ==========================================

def extract_morphological_features(ecg_clean, waves, r_peaks, sampling_rate):
    """
    Extract morphological features from P-QRS-T complex.
    """
    features = {}

    def _wave_indices(name):
        values = waves.get(name, [])
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)].astype(int)
        return values[(values >= 0) & (values < len(ecg_clean))]

    def _paired_interval(start_name, end_name):
        start = np.asarray(waves.get(start_name, []), dtype=float)
        end = np.asarray(waves.get(end_name, []), dtype=float)
        n = min(len(start), len(end))
        if n == 0:
            return np.array([])
        start = start[:n]
        end = end[:n]
        valid = np.isfinite(start) & np.isfinite(end)
        intervals = end[valid] - start[valid]
        return intervals[intervals > 0]
    
    # Global Signal Features
    features['Signal_Mean'] = np.mean(ecg_clean)
    features['Signal_Std'] = np.std(ecg_clean)
    features['Signal_Skewness'] = skew(ecg_clean)
    features['Signal_Kurtosis'] = kurtosis(ecg_clean)
    features['Signal_Min'] = np.min(ecg_clean)
    features['Signal_Max'] = np.max(ecg_clean)
    features['Signal_Range'] = features['Signal_Max'] - features['Signal_Min']

    # P Wave Features
    if "ECG_P_Peaks" in waves:
        p_peaks = _wave_indices("ECG_P_Peaks")
        if len(p_peaks) > 0:
            features['P_Amplitude_Mean'] = np.mean(ecg_clean[p_peaks])
            features['P_Amplitude_Std'] = np.std(ecg_clean[p_peaks])
            features['P_Wave_Amplitude'] = features['P_Amplitude_Mean']

    # T Wave Features
    if "ECG_T_Peaks" in waves:
        t_peaks = _wave_indices("ECG_T_Peaks")
        if len(t_peaks) > 0:
            features['T_Amplitude_Mean'] = np.mean(ecg_clean[t_peaks])
            features['T_Amplitude_Std'] = np.std(ecg_clean[t_peaks])
            features['T_Wave_Amplitude'] = features['T_Amplitude_Mean']

    # ST Segment Level (J-point + 60 ms relative to the local PR baseline)
    if "ECG_R_Offsets" in waves and "ECG_P_Onsets" in waves:
        r_offsets = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
        p_onsets = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
        st_offset = int(round(0.06 * sampling_rate))
        n = min(len(r_offsets), len(p_onsets))
        st_levels = []
        for p_onset, r_offset in zip(p_onsets[:n], r_offsets[:n]):
            if not np.isfinite(p_onset) or not np.isfinite(r_offset):
                continue
            p_onset = int(p_onset)
            r_offset = int(r_offset)
            st_idx = r_offset + st_offset
            if 0 <= p_onset < len(ecg_clean) and 0 <= st_idx < len(ecg_clean):
                baseline_start = max(0, p_onset - int(round(0.04 * sampling_rate)))
                baseline = np.mean(ecg_clean[baseline_start:p_onset + 1])
                st_levels.append(ecg_clean[st_idx] - baseline)
        if len(st_levels) > 0:
            features['ST_Segment_Level'] = np.mean(st_levels)

    # PR Interval
    if "ECG_P_Offsets" in waves and "ECG_R_Onsets" in waves:
        pr_intervals = _paired_interval("ECG_P_Offsets", "ECG_R_Onsets")
        if len(pr_intervals) > 0:
            features['PR_Interval_Mean'] = np.mean(pr_intervals) / sampling_rate
            features['PR_Interval_Std'] = np.std(pr_intervals) / sampling_rate

    # QRS Duration
    if "ECG_R_Onsets" in waves and "ECG_R_Offsets" in waves:
        qrs_durations = _paired_interval("ECG_R_Onsets", "ECG_R_Offsets")
        if len(qrs_durations) > 0:
            features['QRS_Duration_Mean'] = np.mean(qrs_durations) / sampling_rate
            features['QRS_Duration_Std'] = np.std(qrs_durations) / sampling_rate
            features['QRS_Duration'] = features['QRS_Duration_Mean']

    # QT and JT Intervals
    if "ECG_T_Offsets" in waves and "ECG_R_Onsets" in waves:
        qt_intervals = _paired_interval("ECG_R_Onsets", "ECG_T_Offsets")
        if len(qt_intervals) > 0:
            features['QT_Interval_Mean'] = np.mean(qt_intervals) / sampling_rate
            features['QT_Interval_Std'] = np.std(qt_intervals) / sampling_rate
            features['QT_Interval'] = features['QT_Interval_Mean']

    if "ECG_T_Offsets" in waves and "ECG_S_Peaks" in waves:
        jt_intervals = _paired_interval("ECG_S_Peaks", "ECG_T_Offsets")
        if len(jt_intervals) > 0:
            features['JT_Interval_Mean'] = np.mean(jt_intervals) / sampling_rate
            features['JT_Interval_Std'] = np.std(jt_intervals) / sampling_rate
            features['JT_Interval'] = features['JT_Interval_Mean']

    return features

def extract_hrv_features(r_peaks, sampling_rate):
    """
    Extract Time, Frequency, and Non-linear HRV features.
    """
    features = {}
    
    # 1. Time-Domain HRV
    try:
        hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
        for col in hrv_time.columns:
            features[col] = hrv_time[col].values[0]
        rr_ms = np.diff(r_peaks) / sampling_rate * 1000
        if len(rr_ms) > 1:
            counts, _ = np.histogram(rr_ms, bins="auto")
            if len(counts) > 0 and np.max(counts) > 0:
                features['HRV_Triangular_Index'] = len(rr_ms) / np.max(counts)
    except Exception as e:
        print(f"Warning: Time-domain HRV failed: {e}")

    # 2. Frequency-Domain HRV
    try:
        hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate)
        for col in hrv_freq.columns:
            features[col] = hrv_freq[col].values[0]
    except Exception as e:
        print(f"Warning: Frequency-domain HRV failed: {e}")

    # 3. Non-Linear and Complexity HRV (includes Entropy)
    try:
        hrv_nonlin = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate)
        for col in hrv_nonlin.columns:
            features[col] = hrv_nonlin[col].values[0]
    except Exception as e:
        print(f"Warning: Non-linear HRV failed: {e}")
        
    return features

def extract_coupling_features(ecg_clean, r_peaks, sampling_rate):
    """
    Extract Cardiorespiratory Coupling Features (RSA, EDR).
    """
    features = {}
    try:
        # Extract ECG-Derived Respiration (EDR)
        rsp_signal = nk.ecg_rsp(ecg_rate=nk.signal_rate(r_peaks, sampling_rate=sampling_rate, desired_length=len(ecg_clean)), sampling_rate=sampling_rate)
        ecg_rate = nk.signal_rate(r_peaks, sampling_rate=sampling_rate, desired_length=len(ecg_clean))
        features['EDR'] = np.mean(rsp_signal)
        features['EDR_Mean'] = np.mean(rsp_signal)
        features['EDR_Std'] = np.std(rsp_signal)
        if np.std(ecg_rate) > 0 and np.std(rsp_signal) > 0:
            ecg_phase = np.angle(hilbert(ecg_rate - np.mean(ecg_rate)))
            rsp_phase = np.angle(hilbert(rsp_signal - np.mean(rsp_signal)))
            features['Phase_Sync_ECG_Resp'] = np.abs(np.mean(np.exp(1j * (ecg_phase - rsp_phase))))
        
        # Calculate RSA
        rsa = nk.hrv_rsa(r_peaks, rsp_signal, sampling_rate=sampling_rate)
        if isinstance(rsa, pd.DataFrame) and 'RSA_P2T' in rsa.columns:
            features['RSA_Amplitude'] = rsa['RSA_P2T'].values[0]
        elif isinstance(rsa, dict) and 'RSA_P2T' in rsa:
            features['RSA_Amplitude'] = rsa['RSA_P2T']
    except Exception as e:
        print(f"Warning: Cardiorespiratory coupling features failed: {e}")
        
    return features

def process_file_and_extract_features(file_path, sampling_rate=250):
    """
    Load data, preprocess, and extract all comprehensive ECG features.
    """
    # 1. Load Data
    try:
        if file_path.endswith('.mat'):
            mat_data = scipy.io.loadmat(file_path)
            # Assuming data is under 'data' key or first array
            key = 'data' if 'data' in mat_data else [k for k in mat_data.keys() if not k.startswith('__')][0]
            ecg_raw = mat_data[key].flatten()
        else:
            raise ValueError("Unsupported file format. Please implement specific data loading.")
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}")

    if len(ecg_raw) < sampling_rate * 5: # At least 5 seconds of data needed
        raise ValueError("ECG signal is too short (< 5s).")

    # 2. Preprocess Signal
    ecg_clean = preprocess_ecg(ecg_raw, sampling_rate=sampling_rate)

    # 3. Detect R-Peaks and Delineate Waves
    try:
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
        
        if len(r_peaks) < 5:
            raise ValueError("Not enough R-peaks detected.")
            
        _, waves = nk.ecg_delineate(ecg_clean, r_peaks, sampling_rate=sampling_rate, method="dwt")
    except Exception as e:
        raise RuntimeError(f"Wave delineation failed: {e}")

    # 4. Extract Features
    features = {}
    
    # Static info
    features['RHR'] = 60 / np.mean(np.diff(r_peaks) / sampling_rate)
    
    # Merge distinct feature groups
    features.update(extract_morphological_features(ecg_clean, waves, r_peaks, sampling_rate))
    features.update(extract_hrv_features(r_peaks, sampling_rate))
    features.update(extract_coupling_features(ecg_clean, r_peaks, sampling_rate))
    
    return features

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Standardized ECG Feature Extraction for Cognitive Impairment Assessment")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input (.mat) files")
    parser.add_argument('--output_file', type=str, default='ecg_features_output.csv', help="Output CSV file path")
    parser.add_argument('--fs', type=int, default=250, help="Sampling frequency of the ECG data")
    
    args = parser.parse_args()
    
    input_files = glob.glob(os.path.join(args.input_dir, '*.mat'))
    if not input_files:
        print(f"No .mat files found in {args.input_dir}")
        return

    all_features = []
    
    for file_path in input_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}...")
        try:
            feats = process_file_and_extract_features(file_path, sampling_rate=args.fs)
            feats['File_Name'] = filename
            # Move File_Name to the first column
            feats = {'File_Name': feats.pop('File_Name'), **feats}
            all_features.append(feats)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if all_features:
        df_out = pd.DataFrame(all_features)
        df_out.to_csv(args.output_file, index=False)
        print(f"Successfully processed {len(all_features)} files. Results saved to {args.output_file}")
    else:
        print("No valid features extracted from any file.")

if __name__ == "__main__":
    main()
