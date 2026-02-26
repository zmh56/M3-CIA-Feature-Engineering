# EEG Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of EEG (Electroencephalography) features extracted for cognitive impairment assessment in the M3-CIA framework. These features capture nonlinear complexity, information entropy, spectral power and dynamics, time-domain statistics, waveform morphology, and (optionally) multi-channel connectivity—serving as objective biomarkers for detecting and monitoring cognitive impairment.

## Overview

EEG features are extracted from cognitive tasks, capturing neural activity patterns associated with cognitive processing. The feature set includes both classic measures (e.g., spectral edge frequency, peak alpha frequency, coherence) and emerging biomarkers (e.g., multiscale entropy, aperiodic exponent). Features are organized into six categories aligned with their physiological and cognitive relevance.

## Feature Categories and Descriptions

### 1. Nonlinear Complexity

These measures quantify the chaotic and complex nature of neural dynamics, which are altered in cognitive impairment.

#### Lyapunov Exponent (`lyapunov_exponent`)
- **Definition**: Measures system chaos and sensitivity to initial conditions
- **Formula**: $\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta x(t)|}{|\delta x(0)|}$
- **Computation**: Quantifies the rate of divergence of nearby trajectories in phase space
- **Relevance to Cognitive Impairment**: Reflects neural system flexibility and adaptability in cognitive processing; reduced chaos may indicate pathological brain states and dementia progression

#### Higuchi Fractal Dimension (`fractal_dimension` / `HFD`)
- **Definition**: Quantifies signal self-similarity and "jaggedness"
- **Formula**: $D = \frac{\log(L(k))}{\log(k)}$ where $L(k)$ is the average length of the curve at scale $k$
- **Computation**: Based on the relationship between signal length and measurement scale
- **Relevance to Cognitive Impairment**: Indicates the complexity of brain functional geometry and damage severity; fractal dimension is reduced in Alzheimer's disease

#### Lempel-Ziv Complexity (`lz_complexity`)
- **Definition**: Counts the number of distinct substrings in a binary sequence
- **Formula**: $C_{LZ} = \frac{c(n)}{n/\log_2(n)}$ where $c(n)$ is the number of distinct patterns
- **Algorithm**: Converts signal to binary sequence via median threshold, then computes pattern complexity
- **Relevance to Cognitive Impairment**: Measures the richness and diversity of underlying neural patterns; LZ complexity is decreased in mild cognitive impairment

#### Petrosian Fractal Dimension (`petrosian_fd`)
- **Definition**: Estimates fractal dimension from the ratio of turning points to zero crossings in the binary derivative sequence
- **Computation**: Converts signal to binary sequence, counts distinct patterns; $D_P \propto \log N / (\log N + \log(N/N_{\delta}))$ where $N_{\delta}$ relates to sign-change statistics
- **Relevance to Cognitive Impairment**: Alternative to Higuchi FD; faster computation; reduced in AD; complements HFD for complexity assessment

### 2. Information Entropy

Entropy measures capture the regularity, irregularity, and information content of neural signals, which are altered in cognitive disorders.

#### Approximate Entropy (`apen`)
- **Definition**: Quantifies time-series regularity (lower = more regular)
- **Formula**: $ApEn(m,r,N) = \phi^m(r) - \phi^{m+1}(r)$ where $\phi^m(r) = \frac{1}{N-m+1}\sum_{i=1}^{N-m+1}\ln C_i^m(r)$
- **Relevance to Cognitive Impairment**: Captures the degree of neural signal irregularity associated with information processing; ApEn is decreased in Alzheimer's disease

#### Sample Entropy (`sampen`)
- **Definition**: Robust ApEn variant, less dependent on data length
- **Formula**: $SampEn(m,r,N) = -\ln\frac{A}{B}$ where $A$ and $B$ are template match counts
- **Relevance to Cognitive Impairment**: Indicates complexity of neural dynamics and robustness of information encoding; more reliable for cognitive assessment than ApEn

#### Permutation Entropy (`pe`)
- **Definition**: Complexity measure based on ordinal patterns
- **Formula**: $PE = -\sum_{\pi} p(\pi) \log p(\pi)$ where $\pi$ represents ordinal patterns
- **Relevance to Cognitive Impairment**: Characterizes temporal pattern diversity relevant to cognitive state transitions; effective for detecting cognitive changes

#### Fuzzy Entropy (`fuzzy_entropy`)
- **Definition**: Uses fuzzy membership functions, suited for nonstationary signals
- **Computation**: Similar to SampEn but with fuzzy similarity instead of Heaviside function
- **Relevance to Cognitive Impairment**: Provides insight into tolerance of neural activity to uncertainty and noise; robust for EEG with artifacts

#### Differential Entropy (`differential_entropy`)
- **Definition**: Based on signal amplitude distribution (Gaussian assumption)
- **Formula**: $H_{diff} = \frac{1}{2}\log(2\pi e \sigma^2)$ where $\sigma^2$ is signal variance
- **Relevance to Cognitive Impairment**: Represents variability in neural information content across cognitive states; altered in cognitive disorders

#### Spectral Entropy (`spectral_entropy`)
- **Definition**: Entropy of the normalized power spectral density (uniform distribution = maximum entropy)
- **Formula**: $H_{spec} = -\sum_{i} p_i \log p_i$ where $p_i = P(f_i)/\sum_j P(f_j)$
- **Relevance to Cognitive Impairment**: Measures complexity of frequency distribution; reduced in dementia; classic qEEG measure for cognitive assessment

#### Multiscale Sample Entropy (`multiscale_entropy` / `MSE`)
- **Definition**: Sample entropy computed across multiple temporal scales via coarse-graining
- **Formula**: For scale $\tau$, coarse-grained series $y_j^{(\tau)} = \frac{1}{\tau}\sum_{i=(j-1)\tau+1}^{j\tau} x_i$; MSE = SampEn at each scale
- **Relevance to Cognitive Impairment**: Characterizes complexity across temporal scales; AD shows reduced complexity at small scales and altered patterns at large scales; can distinguish healthy vs. MCI vs. AD severity stages; robust biomarker for longitudinal monitoring

### 3. Spectral Power and Ratios

Band power and spectral ratios reflect the balance of neural oscillations across frequency bands, which are hallmark biomarkers of cognitive impairment.

#### Delta Power (`band_energy_1`)
- **Frequency Range**: 0.5–4 Hz (slow-wave activity)
- **Formula**: $E_{\delta} = \sum_{f=0.5}^{4} P(f)$ where $P(f)$ is power spectral density
- **Relevance to Cognitive Impairment**: Reflects baseline neural regulation and large-scale cortical dynamics; increased delta activity is common in severe cognitive impairment

#### Theta Power (`band_energy_2`)
- **Frequency Range**: 4–8 Hz (related to relaxation/light sleep)
- **Formula**: $E_{\theta} = \sum_{f=4}^{8} P(f)$
- **Relevance to Cognitive Impairment**: Associated with attention modulation and working memory processes; theta power changes are early markers of cognitive decline

#### Alpha Power (`band_energy_3`)
- **Frequency Range**: 8–12 Hz (normal adult eyes-closed rhythm)
- **Formula**: $E_{\alpha} = \sum_{f=8}^{12} P(f)$
- **Relevance to Cognitive Impairment**: Linked to cortical inhibition and functional network coordination; alpha power reduction is characteristic of Alzheimer's disease

#### Beta Power (`band_energy_4`)
- **Frequency Range**: 12–30 Hz (active thinking and attention)
- **Formula**: $E_{\beta} = \sum_{f=12}^{30} P(f)$
- **Relevance to Cognitive Impairment**: Reflects engagement of attentional resources and task-related processing; beta activity changes indicate cognitive load variations

#### Gamma Power (`band_energy_5`)
- **Frequency Range**: >30 Hz (information processing and neural synchrony)
- **Formula**: $E_{\gamma} = \sum_{f>30} P(f)$
- **Relevance to Cognitive Impairment**: Indicates fine-scale integration and higher-order cognitive functions; gamma activity reduction affects cognitive integration

#### Alpha/Theta Ratio (`ratio1`)
- **Definition**: Ratio of α to θ power
- **Formula**: $R_{\alpha/\theta} = \frac{E_{\alpha}}{E_{\theta}}$
- **Relevance to Cognitive Impairment**: Serves as an index of balance between inhibitory and attentional control systems; reduced ratio may indicate cognitive dysfunction

#### Beta/Theta Ratio (`ratio2`)
- **Definition**: Ratio of β to θ power
- **Formula**: $R_{\beta/\theta} = \frac{E_{\beta}}{E_{\theta}}$
- **Relevance to Cognitive Impairment**: Highlights shifts in excitation–inhibition balance relevant to cognition; important marker for cognitive assessment

#### Spectral Dynamics (Classic & Emerging)

##### Peak Alpha Frequency (`peak_alpha_frequency` / `PAF`)
- **Definition**: Dominant frequency within the alpha band (7–13 Hz)
- **Formula**: $PAF = \arg\max_{f \in [7,13]} P(f)$
- **Relevance to Cognitive Impairment**: Brain-wide slowing of alpha rhythms in MCI; reduced PAF correlates with cognitive function across PD, post-stroke cognitive impairment, and AD; predictive of cognitive decline over 10 years; strong discriminative power (AUC ~0.77)

##### Spectral Edge Frequency (`spectral_edge_frequency` / `SEF`)
- **Definition**: Frequency below which a specified percentage (e.g., 95%) of total power is contained
- **Formula**: $SEF_{95} = f$ such that $\int_0^f P(\nu) d\nu = 0.95 \int_0^{f_s/2} P(\nu) d\nu$
- **Relevance to Cognitive Impairment**: Distinguishes dementia patients from healthy controls; reflects overall spectral slowing

##### Mean/Median Frequency (`mean_frequency`, `median_frequency`)
- **Definition**: Center of mass of the power spectrum (mean) or frequency dividing spectrum into equal halves (median)
- **Formula**: $f_{mean} = \frac{\sum f \cdot P(f)}{\sum P(f)}$; $f_{median}$: median of cumulative power
- **Relevance to Cognitive Impairment**: Global shift toward lower frequencies indicates cognitive slowing; sensitive to AD progression

##### Aperiodic Exponent (`aperiodic_exponent` / `1/f slope`)
- **Definition**: Slope of the 1/f-like (aperiodic) component in the power spectrum
- **Formula**: $\log P(f) \approx - \chi \cdot \log f + c$; exponent $\chi$ reflects excitation–inhibition balance
- **Relevance to Cognitive Impairment**: Reflects neural population dynamics; combined with IAPF, "mismatched" values predict greater cognitive decline; emerging biomarker in resting-state EEG

### 4. Time-Domain Statistics

These features characterize signal amplitude, higher-order dynamics, and distributional properties without full spectral analysis.

#### RMS Amplitude (`rms_amplitude`)
- **Definition**: Root Mean Square of signal amplitude
- **Formula**: $RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$
- **Relevance to Cognitive Impairment**: Characterizes overall signal power and amplitude variability related to neural activity

#### First-Order Differences (`diff1_mean`, `diff1_std`)
- **Definition**: Mean and standard deviation of $\Delta x_i = x_{i+1} - x_i$
- **Formula**: $\mu_{diff1} = \frac{1}{N-1}\sum_{i=1}^{N-1} \Delta x_i$, $\sigma_{diff1} = \sqrt{\frac{1}{N-2}\sum_{i=1}^{N-1} (\Delta x_i - \mu_{diff1})^2}$
- **Relevance to Cognitive Impairment**: Captures higher-order signal dynamics linked to temporal processing capacity; reflects rate of change in neural activity

#### Second-Order Differences (`diff2_mean`, `diff2_std`)
- **Definition**: Mean and standard deviation of second-order differences (acceleration)
- **Formula**: $\Delta^2 x_i = x_{i+2} - 2x_{i+1} + x_i$
- **Relevance to Cognitive Impairment**: Reflects acceleration in neural activity; may indicate cognitive flexibility

#### Third-Order Differences (`diff3_mean`, `diff3_std`)
- **Definition**: Mean and standard deviation of third-order differences
- **Relevance to Cognitive Impairment**: Captures higher-order signal dynamics; may indicate complex cognitive processing

#### Skewness (`skewness`)
- **Definition**: Measures asymmetry of amplitude distribution
- **Formula**: $\gamma_1 = \frac{E[(x-\mu)^3]}{\sigma^3}$
- **Relevance to Cognitive Impairment**: Detects deviations from Gaussianity often linked to pathological brain states

#### Kurtosis (`kurtosis`)
- **Definition**: Measures tailedness of amplitude distribution
- **Formula**: $\gamma_2 = \frac{E[(x-\mu)^4]}{\sigma^4} - 3$
- **Relevance to Cognitive Impairment**: Detects deviations from Gaussianity; altered in pathological brain states

#### Hjorth Parameters (`hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`)
- **Definition**: Activity (variance), Mobility (dominant frequency proxy), Complexity (bandwidth proxy)
- **Formula**: Activity $= \sigma^2$; Mobility $= \sigma(\dot{x})/\sigma(x)$; Complexity $= \sigma(\ddot{x})/\sigma(\dot{x}) / \text{Mobility}$
- **Relevance to Cognitive Impairment**: Describes spectral properties via time-domain variances; sensitive to slowing in cognitive impairment

### 5. Waveform Morphology

These features capture transient and local properties of the EEG waveform.

#### Peak Amplitude (`peak_amplitude`)
- **Definition**: Extreme voltage values in a window
- **Formula**: $A_{peak} = \max(|x_i|)$
- **Relevance to Cognitive Impairment**: Associated with transient neural synchrony magnitude; amplitude changes indicate neural excitability alterations

#### Zero-Crossing Rate (`zcr`)
- **Definition**: Rate of signal sign-changes per second
- **Formula**: $ZCR = \frac{1}{N-1}\sum_{i=1}^{N-1} \mathbb{1}[\text{sign}(x_{i+1}) \neq \text{sign}(x_i)]$
- **Relevance to Cognitive Impairment**: Proxy for dominant frequency shifts without full spectral analysis

#### Line Length (`line_length`)
- **Definition**: Sum of absolute differences between consecutive points
- **Formula**: $LL = \sum_{i=1}^{N-1} |x_{i+1} - x_i|$
- **Relevance to Cognitive Impairment**: Sensitive to amplitude and frequency changes, reflecting signal "busyness"; useful for detecting state transitions

#### Teager-Kaiser Energy (`teager_kaiser_energy`)
- **Definition**: Instantaneous energy operator capturing amplitude and frequency modulation
- **Formula**: $TKE_i = x_i^2 - x_{i-1} \cdot x_{i+1}$
- **Relevance to Cognitive Impairment**: Tracks transient energy changes; sensitive to oscillatory bursts; useful for detecting abnormal neural synchrony in cognitive tasks

### 6. Multi-Channel Features (Optional)

These features require bilateral or multi-electrode recordings. They capture inter-hemispheric and network-level changes characteristic of cognitive impairment.

#### Brain Symmetry Index (`brain_symmetry_index` / `BSI`)
- **Definition**: Quantifies inter-hemispheric asymmetry between homologous electrode pairs
- **Formula**: $BSI = \frac{1}{N}\sum_{i=1}^{N} \frac{|P_L(f_i) - P_R(f_i)|}{P_L(f_i) + P_R(f_i)}$ for left/right power spectra
- **Relevance to Cognitive Impairment**: Increased asymmetry in dementia; AD shows rostral dominance in complexity, FTD shows caudal dominance; combined BSI with temporal measures improves asymmetry detection

#### Spectral Coherence (`coherence`)
- **Definition**: Frequency-domain correlation between two channels (squared magnitude of cross-spectrum normalized by power)
- **Formula**: $Coh_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}$
- **Relevance to Cognitive Impairment**: Reduced alpha-band coherence is the most consistent finding in AD and MCI; reflects disrupted neural coupling and "disconnection syndrome"; theta coherence may increase as compensatory mechanism

#### Phase-Lag Index (`phase_lag_index` / `PLI`)
- **Definition**: Asymmetry of phase differences between channels; insensitive to volume conduction
- **Formula**: $PLI = |E[\text{sign}(\Delta\phi)]|$
- **Relevance to Cognitive Impairment**: Robust connectivity measure; reduced in AD/MCI; complements coherence for effective connectivity assessment

#### EEG Microstates (`microstate_duration`, `microstate_occurrence`, `microstate_coverage`)
- **Definition**: Quasi-stable topographical patterns reflecting transient brain states (typically 4 canonical classes: A–D)
- **Computation**: Cluster global field power maps; extract duration, occurrence, coverage, and transition probabilities
- **Relevance to Cognitive Impairment**: Temporal dynamics of microstates vary in AD; altered duration and transition patterns; emerging biomarker for characterizing cortical dynamics in cognitive disorders

## Implementation Pipeline

### Data Preprocessing
1. **Sampling Rate**: Typically 250 Hz or higher for adequate frequency resolution
2. **Filtering**: 
   - High-pass filter (>0.5 Hz) to remove DC drift
   - Low-pass filter (<100 Hz) to remove muscle artifacts
   - Notch filter (50/60 Hz) to remove line noise
3. **Artifact Removal**: 
   - Independent Component Analysis (ICA)
   - Automatic artifact detection algorithms
   - Manual artifact rejection
4. **Epoch Length**: 1–4 seconds per epoch depending on cognitive task

### Feature Extraction Pipeline
1. **Segmentation**: Divide continuous EEG into task-specific epochs
2. **Feature Computation**: Extract features from each category for each epoch
3. **Aggregation**: Compute statistics (mean, std) across epochs within each task
4. **Normalization**: Apply z-score normalization or min-max scaling
5. **Quality Control**: Handle missing data and outliers

### Quality Control Measures
- **Missing Data**: Interpolation or exclusion of epochs with >20% missing data
- **Outlier Detection**: Identify values >3 standard deviations from mean
- **Consistency Checks**: Verify feature ranges and distributions
- **Signal Quality**: Ensure adequate signal-to-noise ratio (>10 dB)

## Open-Source Code Libraries

### Python Libraries
1. **MNE-Python**: Comprehensive EEG/MEG analysis
   - Repository: https://github.com/mne-tools/mne-python
   - Features: Preprocessing, artifact removal, source localization, connectivity analysis
   - Documentation: https://mne.tools/

2. **PyEEG**: EEG feature extraction
   - Repository: https://github.com/forrestbao/pyeeg
   - Features: Spectral, entropy, and complexity features
   - Paper: Bao, F. S., et al. (2011). PyEEG: an open source python module for EEG/MEG feature extraction. *Computational Intelligence and Neuroscience*, 2011, 406391.

3. **Antropy**: Entropy measures for time series
   - Repository: https://github.com/raphaelvallat/antropy
   - Features: Approximate entropy, sample entropy, permutation entropy, fuzzy entropy, spectral entropy
   - Documentation: https://raphaelvallat.com/antropy/

4. **NeuroKit2**: Comprehensive neurophysiological signal processing
   - Repository: https://github.com/neuropsychology/NeuroKit
   - Features: ECG, EEG, EDA, EMG processing and analysis
   - Documentation: https://neurokit2.readthedocs.io/

### MATLAB Toolboxes
1. **EEGLAB**: MATLAB toolbox for EEG analysis
   - Website: https://sccn.ucsd.edu/eeglab/
   - Features: ICA, artifact removal, spectral analysis, connectivity

2. **FieldTrip**: Advanced EEG/MEG analysis
   - Website: https://www.fieldtriptoolbox.org/
   - Features: Source analysis, connectivity, statistics, real-time processing

3. **SPM**: Statistical Parametric Mapping
   - Website: https://www.fil.ion.ucl.ac.uk/spm/
   - Features: Statistical analysis, source reconstruction, group analysis

### Online Resources and Tutorials
1. **EEG Signal Processing Tutorial**: https://www.mathworks.com/help/signal/examples/analyzing-brain-signals.html
2. **MNE-Python Tutorial**: https://mne.tools/stable/auto_tutorials/index.html
3. **FieldTrip Tutorial**: https://www.fieldtriptoolbox.org/tutorial/
4. **EEGLAB Tutorial**: https://sccn.ucsd.edu/wiki/EEGLAB

## Conclusion

The EEG feature set described in this document provides a robust foundation for cognitive impairment assessment. Features span six categories—nonlinear complexity, information entropy, spectral power and dynamics, time-domain statistics, waveform morphology, and multi-channel connectivity—each capturing distinct aspects of neural activity. The inclusion of both classic measures (peak alpha frequency, spectral edge frequency, coherence, spectral entropy) and emerging biomarkers (multiscale entropy, aperiodic exponent) enables comprehensive characterization of cognitive states. When combined with appropriate signal processing techniques and clinical validation, these features serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders.

The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of brain activity.
