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
    python extract_eeg_features_standard.py --input path/to/signal.mat --fs 250 --output features.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import cumtrapz
import antropy as ant
import scipy.io as sio


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

def calc_lyapunov_exponent(signal, m=3, tau=2, max_t=20, theiler=None):
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
            times.append(k / tau)

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
        except:
            mse.append(np.nan)
    return np.nanmean(mse)

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
    cum_power = cumtrapz(psd, freqs, initial=0)
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

def extract_all_features(signal, fs):
    """
    Run all feature extraction categories and return a flat dictionary.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    features = {}
    
    # 1. Nonlinear Complexity
    features['lyapunov_exponent'] = calc_lyapunov_exponent(signal)
    features['fractal_dimension'] = ant.higuchi_fd(signal)
    features['lz_complexity'] = calc_lz_complexity(signal)
    features['petrosian_fd'] = calc_petrosian_fd(signal)
    
    # 2. Information Entropy
    try:
        features['apen'] = ant.app_entropy(signal)
    except:
        features['apen'] = np.nan
        
    features['sampen'] = ant.sample_entropy(signal)
    features['pe'] = ant.perm_entropy(signal, normalize=True)
    features['fuzzy_entropy'] = calc_fuzzy_entropy(signal)
    features['differential_entropy'] = 0.5 * np.log(2 * np.pi * np.e * np.var(signal) + 1e-12)
    features['multiscale_entropy'] = calc_multiscale_entropy(signal)
    
    # 3. Spectral Power and Ratios
    spectral_feats = calc_spectral_features(signal, fs)
    features.update(spectral_feats)
    
    # 4 & 5. Time-Domain Statistics & Waveform Morphology
    time_feats = calc_time_domain_and_morphology(signal)
    features.update(time_feats)
    
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract EEG Features defined in M3-CIA framework.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .mat file")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV path")
    parser.add_argument("--fs", type=float, default=250, help="Sampling frequency (Hz)")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to process")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    try:
        data = sio.loadmat(args.input)
        if 'data' in data:
            eeg_mat = data['data']
        else:
            eeg_mat = next(v for k, v in data.items() if isinstance(v, np.ndarray))
            
        eeg_mat = np.array(eeg_mat)
        if eeg_mat.ndim == 2 and eeg_mat.shape[0] < eeg_mat.shape[1]:
            eeg_mat = eeg_mat.T
            
        signal = eeg_mat[:, args.channel].flatten()
        signal = signal[~np.isnan(signal)]
        
        print("Extracting features (this may take a moment)...")
        features = extract_all_features(signal, args.fs)
        
        # Save to CSV
        df = pd.DataFrame([features])
        df.to_csv(args.output, index=False)
        print(f"Extraction successful. Saved to {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
