import argparse
import os
import glob
import pandas as pd
import numpy as np
import scipy.io
import neurokit2 as nk
import pywt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, detrend, medfilt, hilbert, iirnotch
from scipy.stats import skew, kurtosis

MIN_HRV_DURATION_SECONDS = 30.0
MIN_NN_INTERVALS = 20
MIN_RR_MS = 300.0
MAX_RR_MS = 1300.0
MAX_NN_CORRECTION_FRACTION = 0.05

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

def lowpass_filter(ecg_signal, cutoff_freq=40, fs=250, order=5):
    """
    Apply zero-phase Butterworth lowpass filter.
    """
    nyquist = fs / 2.0
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='low')
    return filtfilt(b, a, ecg_signal)

def notch_filter(ecg_signal, powerline_freq=50.0, fs=250, quality_factor=30.0):
    """Apply a zero-phase power-line notch when the frequency is below Nyquist."""
    if powerline_freq <= 0 or powerline_freq >= fs / 2:
        return ecg_signal
    b, a = iirnotch(powerline_freq, quality_factor, fs=fs)
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
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(ecg_signal), wavelet.dec_len)
    level = min(level, max_level)
    if level < 1:
        return ecg_signal.copy()
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

def preprocess_ecg(
    ecg_signal,
    sampling_rate=250,
    lowpass_hz=40.0,
    powerline_hz=50.0,
):
    """
    Standard preprocessing pipeline for ECG signals.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if lowpass_hz <= 0 or lowpass_hz >= sampling_rate / 2:
        raise ValueError("Low-pass cutoff must be between 0 and the Nyquist frequency.")
    window_size = int(2.0 * sampling_rate)
    
    # 1. Baseline wander removal
    ecg_bw_removed = remove_baseline_wander(ecg_signal, window_size)
    
    # 2. Power-line and high-frequency noise removal
    ecg_notched = notch_filter(ecg_bw_removed, powerline_freq=powerline_hz, fs=sampling_rate)
    ecg_lp = lowpass_filter(ecg_notched, cutoff_freq=lowpass_hz, fs=sampling_rate)
    
    # 3. Wavelet denoising
    ecg_denoised = wavelet_denoise(ecg_lp)
    
    # 4. Highpass filtering (remove remaining low-frequency trends)
    ecg_clean = highpass_filter(ecg_denoised, cutoff_freq=0.5, fs=sampling_rate)
    
    return ecg_clean


def interpolate_sparse_missing_samples(signal, max_missing_fraction=0.01):
    """Linearly fill sparse non-finite samples without silently accepting poor data."""
    signal = np.asarray(signal, dtype=float).flatten()
    finite = np.isfinite(signal)
    missing_fraction = 1.0 - np.mean(finite)
    if missing_fraction == 0:
        return signal, missing_fraction
    if np.sum(finite) < 2 or missing_fraction > max_missing_fraction:
        raise ValueError(
            f"ECG contains {missing_fraction:.2%} missing samples; "
            f"the allowed maximum is {max_missing_fraction:.2%}."
        )
    indices = np.arange(signal.size)
    signal[~finite] = np.interp(indices[~finite], indices[finite], signal[finite])
    return signal, missing_fraction


def correct_nn_intervals(
    r_peaks,
    sampling_rate,
    min_rr_ms=MIN_RR_MS,
    max_rr_ms=MAX_RR_MS,
    max_correction_fraction=MAX_NN_CORRECTION_FRACTION,
):
    """Correct physiologically implausible NN intervals and rebuild peak positions."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    rr_ms = np.diff(r_peaks) / sampling_rate * 1000.0
    if rr_ms.size < MIN_NN_INTERVALS:
        raise ValueError(
            f"At least {MIN_NN_INTERVALS} NN intervals are required; found {rr_ms.size}."
        )

    valid = np.isfinite(rr_ms) & (rr_ms >= min_rr_ms) & (rr_ms <= max_rr_ms)
    correction_fraction = 1.0 - np.mean(valid)
    if np.sum(valid) < 2:
        raise ValueError("Too few physiologically valid NN intervals for interpolation.")
    if correction_fraction > max_correction_fraction:
        raise ValueError(
            f"NN correction fraction {correction_fraction:.2%} exceeds "
            f"the allowed maximum {max_correction_fraction:.2%}."
        )

    corrected_rr_ms = rr_ms.copy()
    if not np.all(valid):
        interval_index = np.arange(rr_ms.size)
        if np.sum(valid) >= 4:
            interpolator = CubicSpline(
                interval_index[valid],
                rr_ms[valid],
                bc_type='natural',
                extrapolate=True,
            )
            corrected_rr_ms[~valid] = interpolator(interval_index[~valid])
        else:
            corrected_rr_ms[~valid] = np.interp(
                interval_index[~valid],
                interval_index[valid],
                rr_ms[valid],
            )
        corrected_rr_ms = np.clip(corrected_rr_ms, min_rr_ms, max_rr_ms)

    corrected_steps = np.maximum(
        1,
        np.rint(corrected_rr_ms / 1000.0 * sampling_rate).astype(int),
    )
    corrected_peaks = r_peaks[0] + np.concatenate(([0], np.cumsum(corrected_steps)))
    return corrected_peaks, corrected_rr_ms, correction_fraction

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
            features['PR_Interval_Mean'] = np.mean(pr_intervals) / sampling_rate * 1000.0
            features['PR_Interval_Std'] = np.std(pr_intervals) / sampling_rate * 1000.0

    # QRS Duration
    if "ECG_R_Onsets" in waves and "ECG_R_Offsets" in waves:
        qrs_durations = _paired_interval("ECG_R_Onsets", "ECG_R_Offsets")
        if len(qrs_durations) > 0:
            features['QRS_Duration_Mean'] = np.mean(qrs_durations) / sampling_rate * 1000.0
            features['QRS_Duration_Std'] = np.std(qrs_durations) / sampling_rate * 1000.0
            features['QRS_Duration'] = features['QRS_Duration_Mean']

    # QT and JT Intervals
    if "ECG_T_Offsets" in waves and "ECG_R_Onsets" in waves:
        qt_intervals = _paired_interval("ECG_R_Onsets", "ECG_T_Offsets")
        if len(qt_intervals) > 0:
            features['QT_Interval_Mean'] = np.mean(qt_intervals) / sampling_rate * 1000.0
            features['QT_Interval_Std'] = np.std(qt_intervals) / sampling_rate * 1000.0
            features['QT_Interval'] = features['QT_Interval_Mean']

    if "ECG_T_Offsets" in waves and "ECG_R_Offsets" in waves:
        jt_intervals = _paired_interval("ECG_R_Offsets", "ECG_T_Offsets")
        if len(jt_intervals) > 0:
            features['JT_Interval_Mean'] = np.mean(jt_intervals) / sampling_rate * 1000.0
            features['JT_Interval_Std'] = np.std(jt_intervals) / sampling_rate * 1000.0
            features['JT_Interval'] = features['JT_Interval_Mean']

    return features

def extract_hrv_features(r_peaks, sampling_rate, analyzed_duration_seconds):
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
        if analyzed_duration_seconds < 300.0:
            for col in ('HRV_ULF', 'HRV_VLF'):
                if col in features:
                    features[col] = np.nan
    except Exception as e:
        print(f"Warning: Frequency-domain HRV failed: {e}")

    # 3. Non-Linear and Complexity HRV (includes Entropy)
    try:
        hrv_nonlin = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate)
        for col in hrv_nonlin.columns:
            features[col] = hrv_nonlin[col].values[0]
    except Exception as e:
        print(f"Warning: Non-linear HRV failed: {e}")
        
    features['HRV_Analyzed_Duration_Seconds'] = analyzed_duration_seconds
    features['HRV_VLF_Valid'] = int(analyzed_duration_seconds >= 300.0)
    return features

def extract_coupling_features(ecg_clean, r_peaks, sampling_rate):
    """
    Extract Cardiorespiratory Coupling Features (RSA, EDR).
    """
    features = {}
    try:
        # Extract ECG-Derived Respiration (EDR)
        ecg_rate = nk.signal_rate(
            r_peaks,
            sampling_rate=sampling_rate,
            desired_length=len(ecg_clean),
        )
        rsp_signal = nk.ecg_rsp(ecg_rate=ecg_rate, sampling_rate=sampling_rate)
        features['EDR'] = np.mean(rsp_signal)
        features['EDR_Mean'] = np.mean(rsp_signal)
        features['EDR_Std'] = np.std(rsp_signal)
        if np.std(ecg_rate) > 0 and np.std(rsp_signal) > 0:
            ecg_phase = np.angle(hilbert(ecg_rate - np.mean(ecg_rate)))
            rsp_phase = np.angle(hilbert(rsp_signal - np.mean(rsp_signal)))
            features['Phase_Sync_ECG_Resp'] = np.abs(np.mean(np.exp(1j * (ecg_phase - rsp_phase))))
        
        # hrv_rsa expects processed ECG/RSP signal tables, not raw arrays.
        ecg_peak_indicator = np.zeros(len(ecg_clean), dtype=int)
        valid_peaks = np.asarray(r_peaks, dtype=int)
        valid_peaks = valid_peaks[(valid_peaks >= 0) & (valid_peaks < len(ecg_clean))]
        ecg_peak_indicator[valid_peaks] = 1
        ecg_signals = pd.DataFrame({
            'ECG_Rate': ecg_rate,
            'ECG_R_Peaks': ecg_peak_indicator,
        })
        rsp_signals, _ = nk.rsp_process(rsp_signal, sampling_rate=sampling_rate)
        rsa = nk.hrv_rsa(
            ecg_signals,
            rsp_signals=rsp_signals,
            rpeaks={'ECG_R_Peaks': valid_peaks},
            sampling_rate=sampling_rate,
        )
        if isinstance(rsa, pd.DataFrame):
            rsa_values = rsa.iloc[0].to_dict()
        elif isinstance(rsa, dict):
            rsa_values = rsa
        else:
            rsa_values = {}
        for key, value in rsa_values.items():
            if np.isscalar(value):
                features[key] = value
        if 'RSA_P2T_Mean' in rsa_values:
            features['RSA_Amplitude'] = rsa_values['RSA_P2T_Mean']
    except Exception as e:
        print(f"Warning: Cardiorespiratory coupling features failed: {e}")
        
    return features

def process_file_and_extract_features(
    file_path,
    sampling_rate=250,
    lowpass_hz=40.0,
    powerline_hz=50.0,
    max_nn_correction_fraction=MAX_NN_CORRECTION_FRACTION,
):
    """
    Load data, preprocess, and extract all comprehensive ECG features.
    """
    # 1. Load Data
    try:
        if file_path.endswith('.mat'):
            mat_data = scipy.io.loadmat(file_path)
            # Assuming data is under 'data' key or first array
            key = 'data' if 'data' in mat_data else [k for k in mat_data.keys() if not k.startswith('__')][0]
            ecg_raw = np.asarray(mat_data[key], dtype=float).flatten()
        else:
            raise ValueError("Unsupported file format. Please implement specific data loading.")
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}")

    if sampling_rate <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not 0 <= max_nn_correction_fraction < 1:
        raise ValueError("Maximum NN correction fraction must be in [0, 1).")
    minimum_samples = int(np.ceil(sampling_rate * MIN_HRV_DURATION_SECONDS))
    if len(ecg_raw) < minimum_samples:
        raise ValueError(
            f"ECG signal is too short for HRV (< {MIN_HRV_DURATION_SECONDS:.0f}s)."
        )
    ecg_raw, missing_sample_fraction = interpolate_sparse_missing_samples(ecg_raw)

    # 2. Preprocess Signal
    ecg_clean = preprocess_ecg(
        ecg_raw,
        sampling_rate=sampling_rate,
        lowpass_hz=lowpass_hz,
        powerline_hz=powerline_hz,
    )

    # 3. Detect R-Peaks and Delineate Waves
    try:
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
        
        if len(r_peaks) < MIN_NN_INTERVALS + 1:
            raise ValueError(
                f"Not enough R-peaks detected; at least {MIN_NN_INTERVALS + 1} are required."
            )
            
        _, waves = nk.ecg_delineate(ecg_clean, r_peaks, sampling_rate=sampling_rate, method="dwt")
    except Exception as e:
        raise RuntimeError(f"Wave delineation failed: {e}")

    # 4. Correct implausible NN intervals for HRV while retaining original
    # R-peaks for P-QRS-T morphology.
    corrected_r_peaks, corrected_rr_ms, correction_fraction = correct_nn_intervals(
        r_peaks,
        sampling_rate,
        max_correction_fraction=max_nn_correction_fraction,
    )
    corrected_r_peaks = corrected_r_peaks[corrected_r_peaks < len(ecg_clean)]
    if len(corrected_r_peaks) < MIN_NN_INTERVALS + 1:
        raise ValueError("Too few corrected R-peaks remain within the recording.")
    corrected_rr_ms = np.diff(corrected_r_peaks) / sampling_rate * 1000.0
    analyzed_duration_seconds = np.sum(corrected_rr_ms) / 1000.0
    if analyzed_duration_seconds < MIN_HRV_DURATION_SECONDS:
        raise ValueError(
            f"Clean NN duration is {analyzed_duration_seconds:.1f}s; "
            f"at least {MIN_HRV_DURATION_SECONDS:.0f}s is required."
        )

    # 5. Extract Features
    features = {}
    
    # Static info
    features['RHR'] = 60000.0 / np.mean(corrected_rr_ms)
    features['ECG_Sampling_Rate_Hz'] = sampling_rate
    features['ECG_Recording_Duration_Seconds'] = len(ecg_raw) / sampling_rate
    features['ECG_Missing_Sample_Fraction'] = missing_sample_fraction
    features['ECG_Detected_R_Peak_Count'] = len(r_peaks)
    features['ECG_Valid_NN_Count'] = len(corrected_rr_ms)
    features['ECG_NN_Correction_Fraction'] = correction_fraction
    features['ECG_NN_Min_Milliseconds'] = np.min(corrected_rr_ms)
    features['ECG_NN_Max_Milliseconds'] = np.max(corrected_rr_ms)
    features['ECG_Lowpass_Hz'] = lowpass_hz
    features['ECG_Powerline_Notch_Hz'] = (
        powerline_hz if 0 < powerline_hz < sampling_rate / 2 else np.nan
    )
    
    # Merge distinct feature groups
    features.update(extract_morphological_features(ecg_clean, waves, r_peaks, sampling_rate))
    features.update(
        extract_hrv_features(
            corrected_r_peaks,
            sampling_rate,
            analyzed_duration_seconds,
        )
    )
    features.update(extract_coupling_features(ecg_clean, corrected_r_peaks, sampling_rate))
    
    return features

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Standardized ECG Feature Extraction for Cognitive Impairment Assessment")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input (.mat) files")
    parser.add_argument('--output_file', type=str, default='ecg_features_output.csv', help="Output CSV file path")
    parser.add_argument('--fs', type=int, default=250, help="Sampling frequency of the ECG data")
    parser.add_argument(
        '--lowpass_hz',
        type=float,
        default=40.0,
        help="Zero-phase ECG low-pass cutoff in Hz (default: 40)",
    )
    parser.add_argument(
        '--powerline_hz',
        type=float,
        default=50.0,
        help="Power-line notch frequency in Hz; set to 0 to disable (default: 50)",
    )
    parser.add_argument(
        '--max_nn_correction_fraction',
        type=float,
        default=MAX_NN_CORRECTION_FRACTION,
        help="Maximum fraction of NN intervals corrected before rejecting a recording (default: 0.05)",
    )
    
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
            feats = process_file_and_extract_features(
                file_path,
                sampling_rate=args.fs,
                lowpass_hz=args.lowpass_hz,
                powerline_hz=args.powerline_hz,
                max_nn_correction_fraction=args.max_nn_correction_fraction,
            )
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
