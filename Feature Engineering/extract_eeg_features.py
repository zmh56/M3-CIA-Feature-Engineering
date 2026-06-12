"""
Standard EEG Feature Extraction Pipeline
Based on the M3-CIA framework for Cognitive Impairment Assessment.

This script extracts comprehensive EEG features tailored for cognitive tasks,
including:
1. Nonlinear Complexity
2. Information Entropy
3. Spectral Power and Ratios
4. Time-Domain Statistics
5. Waveform Morphology

Usage:
    python extract_eeg_features.py --input path/to/signal.mat --fs 100 --output features.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid
import antropy as ant
import scipy.io as sio

TIME_WINDOW_SECONDS = 2.0
SPECTRAL_WINDOW_SECONDS = 4.0
NONLINEAR_WINDOW_SECONDS = 10.0
WINDOW_OVERLAP = 0.5

NONLINEAR_FEATURE_NAMES = [
    'lyapunov_exponent',
    'fractal_dimension',
    'lz_complexity',
    'petrosian_fd',
]

ENTROPY_FEATURE_NAMES = [
    'apen',
    'sampen',
    'pe',
    'fuzzy_entropy',
    'differential_entropy',
    'multiscale_entropy',
]

SPECTRAL_FEATURE_NAMES = [
    'band_energy_1',
    'band_energy_2',
    'band_energy_3',
    'band_energy_4',
    'band_energy_5',
    'ratio1',
    'ratio2',
    'peak_alpha_frequency',
    'spectral_edge_frequency',
    'mean_frequency',
    'median_frequency',
    'aperiodic_exponent',
    'spectral_entropy',
]

TIME_FEATURE_NAMES = [
    'rms_amplitude',
    'skewness',
    'kurtosis',
    'diff1_mean',
    'diff1_std',
    'diff2_mean',
    'diff2_std',
    'diff3_mean',
    'diff3_std',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'peak_amplitude',
    'zcr',
    'line_length',
    'teager_kaiser_energy',
]


# ==========================================
# 1. Nonlinear Complexity
# ==========================================

def calc_lz_complexity(signal):
    """Lempel-Ziv Complexity"""
    if len(signal) < 2:
        return np.nan
    median_val = np.median(signal)
    binary_seq = ''.join(['1' if s > median_val else '0' for s in signal])
    n = len(binary_seq)
    i, k, lzc = 0, 1, 1
    dictionary = set([binary_seq[0]])
    while i + k <= n:
        sub = binary_seq[i:i + k]
        if sub in dictionary:
            k += 1
        else:
            lzc += 1
            dictionary.add(sub)
            i += k
            k = 1
    return lzc / (n / np.log2(n)) if n > 1 else 0

def calc_petrosian_fd(signal):
    """Petrosian Fractal Dimension"""
    if len(signal) < 2:
        return np.nan
    diff = np.diff(signal)
    N = len(signal)
    num_sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * num_sign_changes)))

def calc_lyapunov_exponent(signal, fs=1.0, m=3, tau=2, max_t=20, theiler=None):
    """Largest Lyapunov exponent using a compact Rosenstein-style estimate."""
    n = len(signal)
    N_embed = n - (m - 1) * tau
    if N_embed <= max_t + 2:
        return np.nan
    
    embedded_data = np.zeros((N_embed, m))
    for i in range(m):
        embedded_data[:, i] = signal[i * tau : i * tau + N_embed]

    if not np.all(np.isfinite(embedded_data)) or np.all(np.std(embedded_data, axis=0) == 0):
        return np.nan

    theiler = max(m * tau, 1) if theiler is None else theiler
    max_t = min(max_t, N_embed - 2)
    tree = cKDTree(embedded_data)
    _, candidate_idx = tree.query(embedded_data, k=min(20, N_embed))

    nearest = np.full(N_embed, -1, dtype=int)
    for i, candidates in enumerate(candidate_idx):
        candidates = np.atleast_1d(candidates)
        valid = candidates[np.abs(candidates - i) > theiler]
        valid = valid[valid + max_t < N_embed]
        if i + max_t < N_embed and len(valid) > 0:
            nearest[i] = valid[0]

    valid_i = np.where(nearest >= 0)[0]
    if len(valid_i) < 2:
        return np.nan

    mean_log_divergence = []
    times = []
    for k in range(max_t + 1):
        i_k = valid_i[valid_i + k < N_embed]
        j_k = nearest[i_k] + k
        mask = j_k < N_embed
        if not np.any(mask):
            continue
        distances = np.linalg.norm(embedded_data[i_k[mask] + k] - embedded_data[j_k[mask]], axis=1)
        distances = distances[distances > 0]
        if len(distances) > 1:
            mean_log_divergence.append(np.mean(np.log(distances)))
            times.append(k / fs)

    if len(times) < 2:
        return np.nan

    coeffs = np.polyfit(times, mean_log_divergence, deg=1)
    return coeffs[0]


# ==========================================
# 2. Information Entropy
# ==========================================

def calc_fuzzy_entropy(x, m=2, r=None, n=2):
    """Fuzzy Entropy"""
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0 or not np.isfinite(r):
        return np.nan
    def phi(m_dim):
        if N - m_dim + 1 <= 0: return 0
        X = np.array([x[i:i + m_dim] for i in range(N - m_dim + 1)])
        X = X - np.mean(X, axis=1, keepdims=True)
        D = squareform(pdist(X, metric='chebyshev'))
        S = np.exp(-(D ** n) / r)
        np.fill_diagonal(S, 0)
        return np.sum(S) / ((N - m_dim + 1) * (N - m_dim) + 1e-12)
    
    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    if phi_m1 <= 0 or phi_m <= 0: return np.nan
    return -np.log(phi_m1 / phi_m)

def calc_multiscale_entropy(signal, max_scale=3):
    """Multiscale Sample Entropy"""
    mse = []
    for scale in range(1, max_scale + 1):
        if scale == 1:
            coarse = signal
        else:
            length = len(signal) - (len(signal) % scale)
            coarse = np.mean(np.reshape(signal[:length], (-1, scale)), axis=1)
        try:
            mse.append(ant.sample_entropy(coarse))
        except Exception:
            mse.append(np.nan)
    return np.nanmean(mse) if np.any(np.isfinite(mse)) else np.nan

def calc_spectral_entropy(psd):
    """Spectral Entropy from PSD"""
    power_sum = np.sum(psd)
    if power_sum <= 0 or not np.isfinite(power_sum):
        return np.nan
    psd_norm = psd / power_sum
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))


# ==========================================
# 3. Spectral Power and Ratios
# ==========================================

def calc_spectral_features(signal, fs):
    """Extract all spectral features, bands, and ratios."""
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    nperseg = int(min(len(signal), fs * 4))
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    total_power = np.trapz(psd, freqs) + 1e-12
    
    # Standard Bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, None)
    }
    
    band_power = {}
    for name, (low, high) in bands.items():
        idx = freqs >= low if high is None else np.logical_and(freqs >= low, freqs <= high)
        if np.sum(idx) > 1:
            band_power[name] = np.trapz(psd[idx], freqs[idx])
        else:
            band_power[name] = 0.0
    
    # Ratios
    ratio_alpha_theta = band_power['alpha'] / (band_power['theta'] + 1e-12)
    ratio_beta_theta = band_power['beta'] / (band_power['theta'] + 1e-12)
    
    # PAF (Peak Alpha Frequency)
    idx_alpha = np.logical_and(freqs >= 7, freqs <= 13)
    paf = freqs[idx_alpha][np.argmax(psd[idx_alpha])] if np.sum(idx_alpha) > 0 else np.nan
    
    # SEF (Spectral Edge Frequency 95%)
    cum_power = cumulative_trapezoid(psd, freqs, initial=0)
    idx_95 = np.where(cum_power >= 0.95 * total_power)[0]
    sef95 = freqs[idx_95[0]] if len(idx_95) > 0 else np.nan
    
    # Mean and Median Frequency
    mean_f = np.trapz(freqs * psd, freqs) / total_power
    idx_50 = np.where(cum_power >= 0.50 * total_power)[0]
    median_f = freqs[idx_50[0]] if len(idx_50) > 0 else np.nan
    
    # Aperiodic Exponent
    idx_fit = np.logical_and(freqs >= 1, freqs <= 40)
    if np.sum(idx_fit) > 1:
        coeffs = np.polyfit(np.log10(freqs[idx_fit]), np.log10(psd[idx_fit] + 1e-12), 1)
        aperiodic_exp = -coeffs[0]
    else:
        aperiodic_exp = np.nan
        
    return {
        'band_energy_1': band_power['delta'],
        'band_energy_2': band_power['theta'],
        'band_energy_3': band_power['alpha'],
        'band_energy_4': band_power['beta'],
        'band_energy_5': band_power['gamma'],
        'ratio1': ratio_alpha_theta,
        'ratio2': ratio_beta_theta,
        'peak_alpha_frequency': paf,
        'spectral_edge_frequency': sef95,
        'mean_frequency': mean_f,
        'median_frequency': median_f,
        'aperiodic_exponent': aperiodic_exp,
        'spectral_entropy': calc_spectral_entropy(psd)
    }


# ==========================================
# 4. & 5. Time-Domain Stats & Morphology
# ==========================================

def calc_time_domain_and_morphology(signal):
    """Extract statistics and waveform morphology."""
    if len(signal) < 4:
        return {
            'rms_amplitude': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'diff1_mean': np.nan,
            'diff1_std': np.nan,
            'diff2_mean': np.nan,
            'diff2_std': np.nan,
            'diff3_mean': np.nan,
            'diff3_std': np.nan,
            'hjorth_activity': np.nan,
            'hjorth_mobility': np.nan,
            'hjorth_complexity': np.nan,
            'peak_amplitude': np.nan,
            'zcr': np.nan,
            'line_length': np.nan,
            'teager_kaiser_energy': np.nan
        }
    diff1 = np.diff(signal, n=1)
    diff2 = np.diff(signal, n=2)
    diff3 = np.diff(signal, n=3)
    
    # Hjorth Parameters
    var_zero = np.var(signal)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if var_d1 > 0 and mobility > 0 else 0
    
    # Morphology
    zcr = np.sum(np.diff(np.sign(signal)) != 0) / (len(signal) - 1)
    line_length = np.sum(np.abs(diff1))
    tke = np.mean(signal[1:-1]**2 - signal[:-2] * signal[2:])
    
    return {
        'rms_amplitude': np.sqrt(np.mean(signal**2)),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
        'diff1_mean': np.mean(diff1),
        'diff1_std': np.std(diff1),
        'diff2_mean': np.mean(diff2),
        'diff2_std': np.std(diff2),
        'diff3_mean': np.mean(diff3),
        'diff3_std': np.std(diff3),
        'hjorth_activity': var_zero,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'peak_amplitude': np.max(np.abs(signal)),
        'zcr': zcr,
        'line_length': line_length,
        'teager_kaiser_energy': tke
    }


# ==========================================
# Orchestrator
# ==========================================

def extract_nonlinear_features(signal, fs):
    """Extract nonlinear and entropy features from one 10-second window."""
    features = {}

    try:
        features['lyapunov_exponent'] = calc_lyapunov_exponent(
            signal,
            fs=fs,
            max_t=max(20, int(round(0.5 * fs))),
            theiler=max(1, int(round(0.5 * fs))),
        )
    except Exception:
        features['lyapunov_exponent'] = np.nan

    for name, function in (
        ('fractal_dimension', ant.higuchi_fd),
        ('lz_complexity', calc_lz_complexity),
        ('petrosian_fd', calc_petrosian_fd),
    ):
        try:
            features[name] = function(signal)
        except Exception:
            features[name] = np.nan

    try:
        features['apen'] = ant.app_entropy(signal)
    except Exception:
        features['apen'] = np.nan

    for name, function in (
        ('sampen', ant.sample_entropy),
        ('pe', lambda x: ant.perm_entropy(x, normalize=True)),
        ('fuzzy_entropy', calc_fuzzy_entropy),
        ('multiscale_entropy', calc_multiscale_entropy),
    ):
        try:
            features[name] = function(signal)
        except Exception:
            features[name] = np.nan

    features['differential_entropy'] = 0.5 * np.log(2 * np.pi * np.e * np.var(signal) + 1e-12)
    return features


def iter_valid_windows(signal, fs, window_seconds, overlap=WINDOW_OVERLAP):
    """Yield complete finite, non-flat windows without bridging missing samples."""
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    if window_samples < 4 or step_samples < 1:
        raise ValueError("Window settings are incompatible with the sampling rate.")

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        window = signal[start:start + window_samples]
        if not np.all(np.isfinite(window)):
            continue
        if np.std(window) <= np.finfo(float).eps:
            continue
        yield window


def count_candidate_windows(signal_length, fs, window_seconds, overlap=WINDOW_OVERLAP):
    """Return the number of complete windows before quality rejection."""
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    if signal_length < window_samples or step_samples < 1:
        return 0
    return 1 + (signal_length - window_samples) // step_samples


def aggregate_window_features(window_features, feature_names):
    """Keep original names as window means and add dispersion summaries."""
    aggregated = {}
    for name in feature_names:
        values = np.asarray(
            [features.get(name, np.nan) for features in window_features],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            aggregated[name] = np.nan
            aggregated[f'{name}_std'] = np.nan
            aggregated[f'{name}_median'] = np.nan
            aggregated[f'{name}_iqr'] = np.nan
            continue
        aggregated[name] = np.mean(values)
        aggregated[f'{name}_std'] = np.std(values)
        aggregated[f'{name}_median'] = np.median(values)
        aggregated[f'{name}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
    return aggregated


def extract_windowed_features(signal, fs):
    """Extract feature groups using the prespecified 2/4/10-second windows."""
    signal = np.asarray(signal, dtype=np.float64).flatten()
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if signal.size == 0:
        raise ValueError("EEG signal is empty.")

    time_windows = list(iter_valid_windows(signal, fs, TIME_WINDOW_SECONDS))
    spectral_windows = list(iter_valid_windows(signal, fs, SPECTRAL_WINDOW_SECONDS))
    nonlinear_windows = list(iter_valid_windows(signal, fs, NONLINEAR_WINDOW_SECONDS))

    time_results = [calc_time_domain_and_morphology(window) for window in time_windows]
    spectral_results = [calc_spectral_features(window, fs) for window in spectral_windows]
    nonlinear_results = [extract_nonlinear_features(window, fs) for window in nonlinear_windows]
    time_candidates = count_candidate_windows(len(signal), fs, TIME_WINDOW_SECONDS)
    spectral_candidates = count_candidate_windows(len(signal), fs, SPECTRAL_WINDOW_SECONDS)
    nonlinear_candidates = count_candidate_windows(len(signal), fs, NONLINEAR_WINDOW_SECONDS)

    features = {}
    features.update(aggregate_window_features(time_results, TIME_FEATURE_NAMES))
    features.update(aggregate_window_features(spectral_results, SPECTRAL_FEATURE_NAMES))
    features.update(
        aggregate_window_features(
            nonlinear_results,
            NONLINEAR_FEATURE_NAMES + ENTROPY_FEATURE_NAMES,
        )
    )

    features.update({
        'eeg_sampling_rate_hz': fs,
        'eeg_recording_duration_seconds': signal.size / fs,
        'eeg_finite_sample_fraction': np.mean(np.isfinite(signal)),
        'eeg_time_window_seconds': TIME_WINDOW_SECONDS,
        'eeg_spectral_window_seconds': SPECTRAL_WINDOW_SECONDS,
        'eeg_nonlinear_window_seconds': NONLINEAR_WINDOW_SECONDS,
        'eeg_window_overlap_fraction': WINDOW_OVERLAP,
        'eeg_valid_time_window_count': len(time_windows),
        'eeg_valid_spectral_window_count': len(spectral_windows),
        'eeg_valid_nonlinear_window_count': len(nonlinear_windows),
        'eeg_candidate_time_window_count': time_candidates,
        'eeg_candidate_spectral_window_count': spectral_candidates,
        'eeg_candidate_nonlinear_window_count': nonlinear_candidates,
        'eeg_valid_time_window_fraction': (
            len(time_windows) / time_candidates if time_candidates else 0.0
        ),
        'eeg_valid_spectral_window_fraction': (
            len(spectral_windows) / spectral_candidates if spectral_candidates else 0.0
        ),
        'eeg_valid_nonlinear_window_fraction': (
            len(nonlinear_windows) / nonlinear_candidates if nonlinear_candidates else 0.0
        ),
    })
    return features


def extract_all_features(signal, fs):
    """Backward-compatible entry point using feature-specific windows."""
    return extract_windowed_features(signal, fs)


def load_eeg_channel(input_path, channel):
    """Load one EEG channel from a MATLAB file without joining missing segments."""
    data = sio.loadmat(input_path)
    arrays = [
        value for key, value in data.items()
        if not key.startswith('__') and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    if 'data' in data and isinstance(data['data'], np.ndarray):
        eeg_mat = data['data']
    elif arrays:
        eeg_mat = arrays[0]
    else:
        raise ValueError("No numeric signal array was found in the MATLAB file.")

    eeg_mat = np.asarray(eeg_mat, dtype=np.float64)
    eeg_mat = np.squeeze(eeg_mat)
    if eeg_mat.ndim == 1:
        if channel != 0:
            raise IndexError("A one-dimensional input only contains channel 0.")
        return eeg_mat
    if eeg_mat.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D EEG array, received shape {eeg_mat.shape}.")
    if eeg_mat.shape[0] < eeg_mat.shape[1]:
        eeg_mat = eeg_mat.T
    if channel < 0 or channel >= eeg_mat.shape[1]:
        raise IndexError(f"Channel {channel} is outside the valid range 0-{eeg_mat.shape[1] - 1}.")
    return eeg_mat[:, channel]


def main():
    parser = argparse.ArgumentParser(description="Extract EEG Features defined in M3-CIA framework.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .mat file")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV path")
    parser.add_argument("--fs", type=float, default=100, help="Sampling frequency (Hz)")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to process")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    try:
        signal = load_eeg_channel(args.input, args.channel)
        
        print("Extracting features (this may take a moment)...")
        features = extract_windowed_features(signal, args.fs)
        
        # Save to CSV
        df = pd.DataFrame([features])
        df.to_csv(args.output, index=False)
        print(f"Extraction successful. Saved to {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
