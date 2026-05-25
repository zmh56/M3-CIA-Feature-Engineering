"""Single-channel EEG denoising script

This script performs preprocessing, CNN-based artifact detection and
Wavelet Quantile Normalization (WQN) denoising on one EEG channel.

Usage:
    python denoise_eeg.py --input path/to/file.mat --channel 0 \
    --model CNN_LSTM_WAVELAT_EOG.pth --output denoised.mat --plot

Dependencies: numpy, scipy, torch, pywt, matplotlib
"""
import argparse
import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample
import pywt
import torch
import torch.nn as nn


sig_len = 500
step_size = 100
vote_rate = 0.4


def butter_bandstop(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], btype='bandstop', fs=fs)
    return filtfilt(b, a, data)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)


def waveget(sig):
    scales = np.arange(1, 64)
    waveletname = 'cmor1.0-0.5'
    cwtmatr, _ = pywt.cwt(sig, scales, waveletname, 1.0)
    power = (np.abs(cwtmatr)) ** 2
    return np.array([float(power[:, i].sum()) for i in range(len(sig))])


def mask_to_intervals(mask):
    if not np.any(mask):
        return []
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(bool), 1)))
    intervals = edges.reshape((len(edges) // 2, 2))
    return [(i, j) for i, j in intervals]


class CNNLSTMModel0(nn.Module):
    def __init__(self):
        super(CNNLSTMModel0, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


# Minimal copy of WaveletQuantileNormalization from methods.py (self-contained)
class WaveletQuantileNormalization:
    def __init__(self, wavelet='sym5', mode='periodization', alpha=1, n=30):
        self.wavelet = wavelet
        self.alpha = alpha
        self.mode = mode
        self.n = n

    def run_single_channel(self, signal, artifacts, fs=None, reference=None):
        restored = signal.copy()

        for n_idx, (i, j) in enumerate(artifacts):
            min_a = 0
            max_b = signal.size
            size = j - i
            level = int(np.log2(size / self.n)) if size > 0 else 0
            if level < 1:
                continue

            ref_size = max(self.n * 2**level, size)
            a = max(min_a, i - ref_size)
            b = min(max_b, j + ref_size)

            coeffs = pywt.wavedec(signal[a:b], self.wavelet, mode=self.mode, level=level)

            for cs in coeffs:
                k = int(np.round(np.log2(b - a) - np.log2(cs.size)))
                ik, jk = np.array([i - a, j - a]) // (2**k)
                refs = [cs[:ik], cs[jk:]]
                if len(refs[0]) == 0 and len(refs[1]) == 0:
                    continue

                order = np.argsort(np.abs(cs[ik:jk])) if jk - ik > 0 else np.array([])
                if order.size == 0:
                    continue
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))

                vals_ref = np.abs(np.concatenate([r for r in refs if len(r) > 0]))
                if vals_ref.size == 0:
                    continue
                ref_order = np.argsort(vals_ref)
                ref_sp = np.linspace(0, len(inv_order), len(ref_order))
                vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

                r = vals_norm / np.abs(cs[ik:jk])
                cs[ik:jk] *= np.minimum(1, r) ** self.alpha

            rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
            # place reconstructed portion back
            start = i - a
            end = j - a
            restored[i:j] = rec[start:end]

        return restored


def predict_artifacts(signal, model, fs=250):
    # notch and bandpass already applied externally
    # build segments using wavelet power
    segments = []
    for i in range(0, len(signal) - sig_len + 1, step_size):
        seg = signal[i:i + sig_len]
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)
        segments.append(waveget(seg_norm))
    if len(segments) == 0:
        return []
    segments = np.array(segments, dtype=float)

    with torch.no_grad():
        model.eval()
        inputs = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.numpy()

    # voting smoothing (window length 5 like original)
    pad = 4
    pv = np.zeros(len(predicted) + pad)
    for i in range(len(pv)):
        start = max(0, i - pad)
        end = min(len(predicted) - 1, i)
        pv[i] = np.mean(predicted[start:end + 1])
    pv = (pv >= vote_rate).astype(int)
    return pv


def run_denoise(input_path, output_path=None, model_path='CNN_LSTM_WAVELAT_EOG.pth', channel=0, fs=250, plot=False):
    data = sio.loadmat(input_path)
    if 'data' in data:
        EEG = data['data']
    else:
        # try to use first matrix in file
        EEG = next(v for k, v in data.items() if isinstance(v, np.ndarray))

    # normalize dimensions
    EEG = np.array(EEG)
    if EEG.ndim == 1:
        EEG = EEG.reshape(-1, 1)
    elif EEG.ndim > 2:
        EEG = np.squeeze(EEG)

    # ensure samples are along the first axis
    if EEG.ndim == 2 and EEG.shape[0] < EEG.shape[1]:
        EEG = EEG.T

    sig = np.array(EEG[:, channel], dtype=float).flatten()
    # drop NaNs at end (robustification from original)
    sig = sig[~np.isnan(sig)]

    sig = butter_bandstop(sig, 49.9, 50.1, fs)
    sig = butter_bandpass_filter(sig, 0.5, 50, fs)

    # load model
    model = CNNLSTMModel0()
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)

    pv = predict_artifacts(sig, model, fs=fs)
    mask = pv == 1
    intervals = mask_to_intervals(mask)
    # convert window intervals to sample intervals
    artis = [[start * step_size, end * step_size] for start, end in intervals]

    wqn = WaveletQuantileNormalization('sym4', n=20)
    restored = wqn.run_single_channel(sig.copy(), artis, fs)

    if output_path:
        sio.savemat(output_path, {'data': restored})

    if plot:
        try:
            import matplotlib.pyplot as plt
            time = np.arange(len(sig)) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(time, sig, label='original', linewidth=0.6)
            plt.plot(time, restored, label='denoised', linewidth=0.6)
            for s, e in artis:
                plt.axvspan(s / fs, e / fs, color='orange', alpha=0.3)
            plt.legend()
            plt.xlabel('Time (s)')
            plt.show()
        except Exception:
            pass

    return restored


def main():
    p = argparse.ArgumentParser(description='Denoise single-channel EEG using CNN+WQN')
    p.add_argument('--input', required=True, help='Input .mat file containing variable `data`')
    p.add_argument('--channel', type=int, default=0, help='Channel index to denoise (0-based)')
    p.add_argument('--model', default='CNN_LSTM_WAVELAT_EOG.pth', help='Path to trained model .pth')
    p.add_argument('--output', default=None, help='Output .mat file to save denoised signal')
    p.add_argument('--fs', type=int, default=250, help='Sampling frequency')
    p.add_argument('--plot', action='store_true', help='Show plot of original and denoised')
    args = p.parse_args()

    out = args.output if args.output else os.path.splitext(args.input)[0] + '_denoised.mat'
    run_denoise(args.input, out, args.model, args.channel, args.fs, args.plot)


if __name__ == '__main__':
    main()
