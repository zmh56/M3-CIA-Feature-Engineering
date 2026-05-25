# Multimodal Sensor Data Structure and Feature Engineering Pipeline

This document clarifies the structure, preprocessing, feature extraction, and intended use of the multimodal data engineering pipeline used for cognitive impairment assessment. It is intended to accompany the released example data, feature description files, and extraction scripts.

The pipeline processes four synchronously acquired modalities: EEG, ECG, speech, and video. Each modality is first handled by a modality-specific preprocessing module and then converted into a tabular feature representation. These features are used as downstream inputs for cognitive impairment modeling and multimodal fusion.

## Repository Components

| Modality | Feature description | Standard extraction script | Raw or intermediate input |
|---|---|---|---|
| EEG | `eeg_features.md` | `eeg/extract_eeg_features.py` | `.mat` signal array, usually key `data` |
| ECG | `ecg_features.md` | `ecg/extract_ecg_features.py` | `.mat` signal array, usually key `data` |
| Speech | `speech_features.md` | `speech/extract_speech_features.py` | `.wav`, `.mp3`, `.flac`, optional transcript `.txt` |
| Video | `video_features.md` | `video/extract_video_features.py` | OpenFace `.csv`, optionally raw `.mp4`/`.avi` if OpenFace is available |



## Overall Data Flow

1. Raw multimodal signals are collected during cognitive tasks.
2. Each sensor stream is stored independently using its native or derived format.
3. Modality-specific preprocessing removes artifacts and normalizes basic signal properties.
4. Feature extraction converts variable-length raw signals into fixed-length numerical descriptors.
5. Extracted feature tables are aligned by participant, task, and recording identifier.
6. The aligned feature matrix is used for statistical analysis and cognitive impairment modeling.

The standard scripts in this repository implement step 3 and step 4 for each modality. Participant labels, demographics, task metadata, and multimodal fusion logic should be joined outside these extraction scripts using file names or study-specific metadata tables.

## Input and Output Semantics

### EEG

Raw EEG is expected as a MATLAB `.mat` file containing a numeric array. The default loader first searches for a `data` key; if absent, it uses the first non-metadata numeric array. A single channel is selected using `--channel`, and the script handles row-vector input by transposing it into a time-by-channel representation.

Example:

```bash
python eeg/extract_eeg_features.py \
  --input eeg/signal/XXXXX.mat \
  --fs 250 \
  --channel 0 \
  --output eeg_features_output.csv
```

Output is a one-row CSV containing nonlinear complexity, entropy, spectral, time-domain, and waveform morphology features. The current standard script extracts single-channel features. Multi-channel connectivity features described as optional in `eeg_features.md`, such as coherence, phase-lag index, and brain symmetry index, require multi-electrode recordings and are not computed by the single-channel standard script.

### ECG

Raw ECG is expected as a MATLAB `.mat` file containing a one-dimensional ECG signal, usually under the `data` key. Batch extraction reads all `.mat` files in `--input_dir`.

Example:

```bash
python ecg/extract_ecg_features.py \
  --input_dir ecg/siganl \
  --fs 250 \
  --output_file ecg_features_output.csv
```

Output is a CSV with one row per recording. Features include resting heart rate, morphology of the P-QRS-T complex, PR/QRS/QT/JT intervals, signal statistics, HRV time-domain features, HRV frequency-domain features, nonlinear HRV metrics, ECG-derived respiration, RSA, and ECG-respiration phase synchronization where estimable.

### Speech

Speech input consists of an audio file and, when available, a transcript with the same basename. Audio files may be `.wav`, `.mp3`, or `.flac`. Transcripts are plain UTF-8 text files stored in `--text_dir`.

Example:

```bash
python speech/extract_speech_features.py \
  --audio_dir speech \
  --text_dir speech/transcripts \
  --output_file speech_features_output.csv
```

The script extracts acoustic and paralinguistic features using OpenSMILE eGeMAPSv02 and supplementary Librosa features. If transcripts are available, it also extracts lexical and semantic features, including word count, type-token ratio, hesitation ratio, semantic coverage, syntax-depth approximation, and optional BERT semantic match score.

BERT semantic matching is optional:

```bash
python speech/extract_speech_features.py \
  --audio_dir speech \
  --text_dir speech/transcripts \
  --use_bert \
  --output_file speech_features_output.csv
```

When `--use_bert` is enabled, the script uses `bert-base-chinese` to embed a canonical Cookie Theft description and compares it with each participant transcript.

### Video

Video features are extracted from OpenFace output CSV files. Each CSV is expected to contain frame-level columns such as `timestamp`, `confidence`, `success`, `AUxx_r`, `AUxx_c`, `pose_Rx`, `pose_Tx`, `gaze_angle_x`, and optionally 68-point landmarks (`x_0` to `x_67`, `y_0` to `y_67`).

Example using existing OpenFace CSV files:

```bash
python video/extract_video_features.py \
  --input_dir video/openface_csv \
  --output_file video_features_output.csv
```

If raw videos are provided and the OpenFace `FeatureExtraction` executable is available, the script can call OpenFace before feature aggregation:

```bash
python video/extract_video_features.py \
  --video_dir video/raw \
  --input_dir video/openface_csv \
  --openface_path FeatureExtraction \
  --output_file video_features_output.csv
```

Output is a CSV with one row per video/OpenFace file. Features include AU intensity summaries, AU activation probability, AU intensity among active frames, expressive entropy, head pose and motor dynamics, nodding frequency, gaze angle statistics, blink rate, fixation duration, landmark velocity, and landmark-based asymmetry.

## Modality-Specific Preprocessing and Feature Extraction

### EEG Pipeline

The EEG feature script processes each selected channel as a one-dimensional time series. In the broader study pipeline, raw EEG signals are band-pass filtered to suppress slow drift and high-frequency noise, dynamically quality-checked using sliding windows, corrected for artifacts using wavelet-based procedures and quantile normalization, and resampled when needed for uniform temporal representation.

The standard feature script extracts:

- Nonlinear dynamics: Lyapunov exponent, Higuchi fractal dimension, Lempel-Ziv complexity, Petrosian fractal dimension.
- Entropy measures: approximate entropy, sample entropy, permutation entropy, fuzzy entropy, differential entropy, multiscale entropy, spectral entropy.
- Spectral measures: delta, theta, alpha, beta, gamma band power; alpha/theta and beta/theta ratios; peak alpha frequency; spectral edge frequency; mean and median frequency; aperiodic exponent.
- Time-domain statistics: RMS amplitude, skewness, kurtosis, first-, second-, and third-order difference statistics, Hjorth parameters.
- Waveform morphology: peak amplitude, zero-crossing rate, line length, and Teager-Kaiser energy.

The implemented EEG bands are:

| Feature | Frequency range |
|---|---|
| `band_energy_1` delta | 0.5-4 Hz |
| `band_energy_2` theta | 4-8 Hz |
| `band_energy_3` alpha | 8-13 Hz |
| `band_energy_4` beta | 13-30 Hz |
| `band_energy_5` gamma | >30 Hz |

These features characterize cortical slowing, neural variability, transient synchrony, and signal irregularity during cognitive processing.

### ECG Pipeline

The ECG script loads raw single-channel ECG, checks minimum duration, removes baseline wander using median filtering and detrending, applies low-pass filtering, performs wavelet denoising, and applies high-pass filtering to remove residual low-frequency trends.

After preprocessing, the script detects R peaks with NeuroKit2 and delineates P, QRS, and T waves using wavelet-based ECG delineation. It then computes:

- Global signal statistics: mean, standard deviation, skewness, kurtosis, min, max, range.
- Morphological features: P and T amplitude, PR interval, QRS duration, QT interval, JT interval, ST segment level.
- Heart-rate measures: resting heart rate and RR-derived metrics.
- HRV time-domain features: NeuroKit2 `hrv_time` outputs, plus triangular index.
- HRV frequency-domain features: NeuroKit2 `hrv_frequency` outputs.
- Nonlinear HRV features: NeuroKit2 `hrv_nonlinear` outputs, including entropy, Poincare, fragmentation, and fractal metrics when supported by the data length.
- Cardiorespiratory coupling: ECG-derived respiration, RSA amplitude, and phase synchronization between heart-rate and EDR signals.

Some HRV features require sufficiently long and stable recordings. When a feature group cannot be reliably estimated, the script logs a warning and continues with the remaining feature groups.

### Speech Pipeline

The speech script resamples audio to 16 kHz and normalizes amplitude for Librosa-based analysis. OpenSMILE eGeMAPSv02 functionals are extracted from the audio file to capture acoustic and paralinguistic properties, including pitch, loudness, jitter, shimmer, HNR, MFCCs, spectral flux, spectral slope, and voiced/unvoiced segment statistics.

Supplementary Librosa features include:

- Spectral centroid.
- Spectral rolloff.
- Spectral bandwidth.
- Spectral contrast.
- Zero-crossing rate.
- F0 range and F0 slope estimated from pitch tracking.

Temporal fluency features are computed by splitting non-silent intervals and measuring pauses:

- Phonation ratio and total speech duration.
- Pause count, total pause duration, average pause duration, and pause ratio.
- Short pauses: 0.15-0.4 seconds.
- Long pauses: 0.4-1.0 seconds.
- Hesitation pauses: >=1.0 second.
- Speech segment count, average segment duration, and maximum segment duration.

If a transcript is provided, the script extracts:

- Total word count.
- Type-token ratio.
- Semantic coverage of Cookie Theft key elements.
- Filled-pause hesitation ratio.
- Syntax-depth approximation based on sentence-level token counts.
- Optional BERT semantic match score against a canonical Cookie Theft description.

The acoustic and semantic features jointly quantify vocal control, speech timing, lexical retrieval, and narrative informativeness.

### Video Pipeline

The video script assumes OpenFace has generated frame-level CSV files. Low-confidence or failed frames are removed when `success` and `confidence` columns are present. The default confidence threshold is 0.7.

The script extracts:

- AU features: mean, standard deviation, and max for selected AU intensity columns.
- AU activation and expressiveness: AUP, AU-specific activation probabilities, AUI, and expressive entropy.
- Head dynamics: pitch/yaw/roll statistics, translation velocity, rotation velocity, combined head velocity, head jerk, and nodding frequency.
- Eye movement: gaze angle statistics, blink rate, fixation duration.
- Facial kinematics: landmark velocity and landmark-based asymmetry index when 68-point landmarks are available.

These features capture facial expressiveness, psychomotor activity, visual attention, and facial motion asymmetry.

## Participant Variability, Sensor Variability, and Noise

The multimodal dataset contains variability from both participants and sensors. This variability is expected and is part of the signal modeling problem.

### Participant-Level Variability

Participant-level sources include:

- Age, sex, education level, and baseline cognitive ability.
- Individual differences in resting neural rhythms, cardiac autonomic tone, speech rate, and facial expressiveness.
- Task engagement, fatigue, anxiety, medication status, and comorbid neurological or cardiovascular conditions.
- Language production differences, dialect, vocabulary choice, and narrative strategy during picture description.

To reduce bias, features should be interpreted statistically across participants and, when possible, normalized within task, session, or participant groups before modeling.

### Sensor and Acquisition Variability

Sensor-level sources include:

- EEG electrode impedance, placement differences, movement artifacts, ocular/muscle artifacts, and line noise.
- ECG electrode contact quality, baseline drift, muscle artifacts, ectopic beats, and respiratory modulation.
- Microphone distance, ambient noise, clipping, room acoustics, and ASR/transcription quality.
- Video illumination, head pose, occlusion, face tracking confidence, frame rate variation, and OpenFace detection reliability.

The scripts include practical safeguards such as confidence filtering for video, baseline/wavelet denoising for ECG, NaN handling for EEG, and silence-based segmentation for speech. However, the final modeling dataset should still undergo feature-level quality control, including missingness checks, outlier inspection, and distributional review.

### Noise Handling by Modality

| Modality | Primary noise sources | Mitigation in pipeline |
|---|---|---|
| EEG | Drift, muscle/ocular artifacts, high-frequency noise, channel quality variation | Band-limited spectral analysis, entropy/complexity safeguards, NaN removal, optional artifact correction in broader preprocessing |
| ECG | Baseline wander, motion artifacts, muscle noise, R-peak errors, ectopic beats | Median baseline removal, detrending, low/high-pass filtering, wavelet denoising, R-peak count checks |
| Speech | Background noise, silence, microphone variability, transcription errors | Resampling, normalization, OpenSMILE functionals, silence segmentation, optional transcript-based semantic analysis |
| Video | Low-confidence tracking, pose/lighting variation, occlusion, frame drops | OpenFace success/confidence filtering, timestamp fallback, AU aggregation, landmark-based summaries |

## Expected Use of Released Data

The released example files are provided to demonstrate:

- Expected file naming and input layout.
- Required raw or intermediate signal structure.
- Feature extraction commands.
- Output feature schemas.
- How modality-specific features map to the feature description tables.

We also provide a representative extracted feature file in `.npz` format (e.g., `review_sample_feature_matrix.npz`) which contains an aligned participant-level feature matrix. The following table briefly summarizes its contents:

| Modality | Description |
|---|---|
| EEG | Mapped EEG features (e.g., statistical, spectral, entropy, complexity) from multiple task blocks. |
| ECG | Mapped ECG/HRV features (e.g., interval measures, frequency-domain outpus) from multiple task blocks. |
| Video | Mapped Video/Facial features (e.g., Action Unit intensity, activation) from specific segments. |
| Speech | Mapped Speech features including acoustic, fluency, vocabulary, and time features. |
| Task scores | Cognitive task score features across different domains. |
| Demographics | Participant baseline variables (e.g., age, gender, education). |
| Final Label | The binary final cognitive impairment label. |
| Domain Targets | Binary targets mapped to various intermediate cognitive domains. |

For full reproducibility and external validation, we recommend releasing: De-identified raw sensor recordings where consent and privacy constraints allow.

## Output Feature Tables and Downstream Modeling

Each standard script outputs a tabular CSV. Rows represent a recording, channel, or file depending on modality; columns represent engineered features. A downstream multimodal integration step should join tables using subject and task identifiers derived from file names or metadata.

Recommended downstream schema:

| Column group | Description |
|---|---|
| `participant_id` | De-identified participant identifier |
| `task_id` | Cognitive task or recording condition |
| `modality` | EEG, ECG, speech, or video |
| `file_name` | Original source file |
| `sampling_rate` / `frame_rate` | Acquisition or processing rate where relevant |
| feature columns | Modality-specific engineered features |
| label columns | Cognitive score, diagnosis, or impairment label |

Missing feature values may occur when a required signal component is absent or unreliable, for example insufficient R peaks in ECG, missing landmarks in video, unavailable transcripts in speech, or too-short EEG segments. These missing values should be handled explicitly during modeling.

## Relationship Between Documentation and Scripts

The four modality description files define the intended feature taxonomy and interpretation:

- `eeg_features.md`: neurophysiological complexity, entropy, spectral, temporal, and waveform features.
- `ecg_features.md`: ECG morphology, HRV, autonomic regulation, cardiorespiratory coupling, and nonlinear dynamics.
- `speech_features.md`: acoustic, prosodic, voice quality, fluency, and semantic features.
- `video_features.md`: facial action units, head motor dynamics, eye movement, and facial kinematics.

The standard scripts provide executable reference implementations for representative feature extraction. Some features described as optional or tool-dependent may require richer raw data, longer recordings, or additional external tools. In particular, EEG multi-channel connectivity requires multi-electrode signals; speech BERT matching requires transcript text and model availability; video raw processing requires OpenFace; and ECG HRV reliability depends strongly on recording length and R-peak quality.

## Summary for Reviewer Response

To address concerns about unclear data format and intended use, the pipeline can be summarized as follows:

We established a modality-specific but structurally consistent processing workflow for EEG, ECG, speech, and video data. Raw sensor streams are stored in modality-appropriate formats, preprocessed to reduce sensor noise and artifacts, and converted into fixed-length feature vectors using documented feature definitions. EEG features quantify neural oscillatory activity, entropy, nonlinear complexity, and waveform morphology. ECG features quantify cardiac morphology, HRV, autonomic regulation, and cardiorespiratory coupling. Speech features quantify acoustic prosody, voice quality, fluency, and semantic content. Video features quantify facial action units, head dynamics, gaze/blink behavior, and facial kinematics. The resulting CSV feature matrices are intended for participant-level cognitive impairment modeling and can be joined using participant/task metadata. The included example files demonstrate format and extraction behavior.
