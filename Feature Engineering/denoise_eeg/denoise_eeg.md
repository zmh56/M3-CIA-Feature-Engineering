# Single-Channel EEG Denoising Pipeline

This directory contains the implementation of our single-channel EEG denoising pipeline (`denoise_eeg.py`), designed specifically to handle severe physiological noise introduced during active cognitive tasks. 

Our experimental pipeline employs **Wavelet Quantile Normalization (WQN)** ([Holcman Lab GitHub](https://github.com/holcman-lab/wavelet-quantile-normalization)) coupled with a custom **CNN-LSTM** artifact segmentation architecture for robust anterior/frontal EEG processing.

## Pipeline Overview

The complete preprocessing procedure is divided into the following five steps:

### Step 1: Frequency-Domain Pre-Filtering
A 0.5–50 Hz band-pass filter is first applied to the single-channel EEG. This suppresses low-frequency baseline drift (typically arising from perspiration or minor sensor displacement) and high-frequency noise. A 50 Hz notch filter is further applied to eliminate power-line interference, preserving the core EEG bands relevant to cognitive assessment.

### Step 2: Dynamic Artifact Segment Detection
Building on an overlapping sliding-window scheme, we developed a **CNN–LSTM architecture** for the automated semantic segmentation of EEG signals. 
- For each window, the standardized signal is transformed via Continuous Wavelet Transform (CWT). The resulting time–frequency energy distribution is used as the model input, enabling precise detection of characteristic low-frequency energy bursts (e.g., eye blinks) and localized spectral distortions (e.g., jaw clenching).
- A 1D Convolutional Neural Network (1D-CNN) extracts local waveform features, while a Long Short-Term Memory (LSTM) layer captures long-range temporal dependencies. The model outputs the probability of the current window being an artifact.
- To ensure temporal continuity, a **sliding multi-window joint voting mechanism** is introduced along the temporal axis to smooth single-pass prediction fluctuations and merge adjacent artifact windows. 
*(Pre-trained weights `CNN_LSTM_WAVELAT_EOG.pth` are provided in this repository).*

### Step 3: WQN-Based Artifact Reconstruction
Each detected artifact segment is subjected to wavelet decomposition. Following the WQN algorithm, the wavelet coefficients of the noisy segment are quantile-normalized against the statistical distribution derived from neighboring clean segments of the same participant. 
- This procedure effectively suppresses high-amplitude artifacts (e.g., ocular or electromyographic activity) while preserving the underlying neurophysiological rhythms essential for cognitive assessment.
- Extreme wavelet coefficients induced by artifacts are remapped onto the physiological distribution of adjacent clean EEG, thereby avoiding the suppression of high-frequency neural activity that conventional broadband filtering would cause.

### Step 4: Downsampling
The reconstructed clean signal can be downsampled (e.g., to 100 Hz), which improves computational efficiency without compromising the principal time–frequency dynamics of the signal.

### Step 5: Final Quality Control and Rejection Criteria
Because extreme motion or muscle artifacts cannot always be fully reconstructed, a strict rejection mechanism is recommended.
- If the proportion of detected artifact segments within a given task trial exceeds a predefined threshold, or if the reconstructed signal still exhibits non-physiological amplitudes, the EEG recording of that task should be labeled as "Missing".
- Missing data can subsequently be handled by modality-level missing-data imputation strategies.

## Usage

**Dependencies:**
`numpy`, `scipy`, `torch`, `pywt`, `matplotlib`

**Command line execution:**
```bash
python denoise_eeg.py \
    --input path/to/file.mat \
    --channel 0 \
    --model CNN_LSTM_WAVELAT_EOG.pth \
    --output denoised.mat \
    --fs 250 \
    --plot
```

### Arguments
- `--input`: Path to the input `.mat` file containing the EEG measurement.
- `--channel`: Channel index to denoise (0-based, default: `0`).
- `--model`: Path to the trained CNN-LSTM model weights (`CNN_LSTM_WAVELAT_EOG.pth`).
- `--output`: Path to save the output `.mat` file block. If not specified, it adopts `_denoised.mat` as suffix.
- `--fs`: Target sampling rate for the given dataset (default `250`).
- `--plot`: Visualize the original and denoised signal along with the auto-detected artifact shaded regions before saving.
