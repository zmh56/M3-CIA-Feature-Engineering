# EEG Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of EEG (Electroencephalography) features extracted for cognitive impairment assessment in the M3-CIA framework. These features capture various aspects of brain activity including statistical properties, frequency content, non-linear dynamics, and complexity measures.

## Overview

EEG features are extracted from cognitive tasks with 27 features per task, capturing the neural activity patterns associated with cognitive processing. These features serve as objective biomarkers for detecting and monitoring cognitive impairment.

## Feature Categories and Descriptions

### 1. Statistical Features

#### Mean (`mean`)
- **Definition**: Average value of the EEG signal
- **Formula**: $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$
- **Physiological Significance**: Reflects the baseline electrical activity of the brain
- **Cognitive Relevance**: Altered baseline activity may indicate cognitive dysfunction
- **Clinical Application**: Baseline shifts are common in neurodegenerative disorders

#### Standard Deviation (`std`)
- **Definition**: Measure of signal variability around the mean
- **Formula**: $\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \mu)^2}$
- **Physiological Significance**: Indicates the degree of neural activity fluctuations
- **Cognitive Relevance**: Reduced variability may suggest decreased neural responsiveness
- **Clinical Application**: Variability changes are associated with cognitive decline

#### Peak Amplitude (`peak_amplitude`)
- **Definition**: Maximum absolute value of the signal
- **Formula**: $A_{peak} = \max(|x_i|)$
- **Physiological Significance**: Represents the strongest neural activity
- **Cognitive Relevance**: May reflect attention and arousal levels
- **Clinical Application**: Amplitude changes indicate neural excitability alterations

### 2. Non-Linear Features

#### Lyapunov Exponent (`lyapunov_exponent`)
- **Definition**: Measure of chaos and sensitive dependence on initial conditions
- **Formula**: $\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta x(t)|}{|\delta x(0)|}$
- **Mathematical Background**: Quantifies the rate of divergence of nearby trajectories in phase space
- **Physiological Significance**: Indicates the chaotic nature of neural dynamics
- **Cognitive Relevance**: Reduced chaos may indicate pathological brain states
- **Clinical Application**: Lyapunov exponent changes are associated with dementia progression

#### Fractal Dimension (`fractal_dimension`)
- **Definition**: Measure of signal complexity using Higuchi method
- **Formula**: $D = \frac{\log(L(k))}{\log(k)}$ where $L(k)$ is the average length of the curve
- **Computation**: Based on the relationship between signal length and measurement scale
- **Physiological Significance**: Reflects the self-similarity and complexity of neural activity
- **Cognitive Relevance**: Altered fractal properties may indicate cognitive impairment
- **Clinical Application**: Fractal dimension is reduced in Alzheimer's disease

#### Lempel-Ziv Complexity (`lz_complexity`)
- **Definition**: Measure of signal complexity based on pattern repetition
- **Algorithm**: Converts signal to binary sequence and computes pattern complexity
- **Formula**: $C_{LZ} = \frac{c(n)}{n/\log_2(n)}$ where $c(n)$ is the number of distinct patterns
- **Physiological Significance**: Quantifies the regularity and predictability of neural activity
- **Cognitive Relevance**: Reduced complexity may indicate cognitive decline
- **Clinical Application**: LZ complexity is decreased in mild cognitive impairment

### 3. Frequency Domain Features

#### Band Energy Features
The EEG spectrum is divided into five frequency bands based on neurophysiological significance:

##### Delta Band Energy (`band_energy_1`)
- **Frequency Range**: 0.5-4 Hz
- **Formula**: $E_{\delta} = \sum_{f=0.5}^{4} P(f)$ where $P(f)$ is power spectral density
- **Physiological Significance**: Associated with deep sleep and unconscious states
- **Cognitive Relevance**: May indicate pathological slowing in dementia
- **Clinical Application**: Increased delta activity is common in severe cognitive impairment

##### Theta Band Energy (`band_energy_2`)
- **Frequency Range**: 4-8 Hz
- **Formula**: $E_{\theta} = \sum_{f=4}^{8} P(f)$
- **Physiological Significance**: Related to memory formation and emotional processing
- **Cognitive Relevance**: Theta activity changes are common in cognitive disorders
- **Clinical Application**: Theta power changes are early markers of cognitive decline

##### Alpha Band Energy (`band_energy_3`)
- **Frequency Range**: 8-12 Hz
- **Formula**: $E_{\alpha} = \sum_{f=8}^{12} P(f)$
- **Physiological Significance**: Associated with relaxed wakefulness and attention
- **Cognitive Relevance**: Alpha rhythm abnormalities are hallmark features of cognitive impairment
- **Clinical Application**: Alpha power reduction is characteristic of Alzheimer's disease

##### Beta Band Energy (`band_energy_4`)
- **Frequency Range**: 12-30 Hz
- **Formula**: $E_{\beta} = \sum_{f=12}^{30} P(f)$
- **Physiological Significance**: Related to active concentration and cognitive processing
- **Cognitive Relevance**: Beta activity changes may reflect cognitive effort
- **Clinical Application**: Beta activity changes indicate cognitive load variations

##### Gamma Band Energy (`band_energy_5`)
- **Frequency Range**: 30-100 Hz
- **Formula**: $E_{\gamma} = \sum_{f=30}^{100} P(f)$
- **Physiological Significance**: Associated with consciousness and cognitive binding
- **Cognitive Relevance**: Gamma activity is crucial for higher cognitive functions
- **Clinical Application**: Gamma activity reduction affects cognitive integration

#### Spectral Ratios

##### Alpha/Theta Ratio (`ratio1（α/θ）`)
- **Definition**: Ratio of alpha to theta band power
- **Formula**: $R_{\alpha/\theta} = \frac{E_{\alpha}}{E_{\theta}} = \frac{\sum_{f=8}^{12} P(f)}{\sum_{f=4}^{8} P(f)}$
- **Physiological Significance**: Reflects the balance between relaxed and active brain states
- **Cognitive Relevance**: Reduced ratio may indicate cognitive dysfunction
- **Clinical Application**: Important biomarker for cognitive assessment

##### Beta/Theta Ratio (`ratio2(β/θ）`)
- **Definition**: Ratio of beta to theta band power
- **Formula**: $R_{\beta/\theta} = \frac{E_{\beta}}{E_{\theta}} = \frac{\sum_{f=12}^{30} P(f)}{\sum_{f=4}^{8} P(f)}$
- **Physiological Significance**: Indicates cognitive activation level
- **Cognitive Relevance**: Important marker for cognitive assessment
- **Clinical Application**: Beta/theta ratio changes reflect cognitive effort

##### Log Alpha/Theta Ratio (`ratio3（log（ratio1））`)
- **Definition**: Logarithm of the alpha/theta ratio
- **Formula**: $R_{log} = \log\left(\frac{E_{\alpha}}{E_{\theta}}\right) = \log\left(\frac{\sum_{f=8}^{12} P(f)}{\sum_{f=4}^{8} P(f)}\right)$
- **Physiological Significance**: Normalized measure for better statistical properties
- **Cognitive Relevance**: More stable measure for longitudinal studies
- **Clinical Application**: Log-transformed ratios provide better statistical properties

### 4. Spectral Entropy Features

#### Spectral Entropy (`spectro_entropy`)
- **Definition**: Entropy of the power spectral density
- **Formula**: $H_{spectral} = -\sum_{i} p_i \log p_i$ where $p_i = \frac{P(f_i)}{\sum_j P(f_j)}$
- **Physiological Significance**: Measures the complexity of frequency distribution
- **Cognitive Relevance**: Reduced spectral entropy may indicate cognitive impairment
- **Clinical Application**: Spectral entropy is decreased in dementia patients

#### Higuchi Spectral Entropy (`HHSE`)
- **Definition**: Spectral entropy computed using Higuchi method
- **Formula**: $H_{HHSE} = -\sum_{k} \frac{L(k)}{\sum_j L(j)} \log \frac{L(k)}{\sum_j L(j)}$
- **Algorithm**: Combines fractal analysis with entropy measures
- **Physiological Significance**: Multi-scale complexity measure
- **Cognitive Relevance**: Advanced measure for cognitive assessment
- **Clinical Application**: HHSE provides robust complexity measures

#### Wavelet Entropy (`we_entropy`)
- **Definition**: Entropy computed from wavelet coefficients
- **Formula**: $H_{wavelet} = -\sum_{j} p_j \log p_j$ where $p_j = \frac{E_j}{\sum_k E_k}$
- **Advantages**: Time-frequency localization
- **Physiological Significance**: Captures both temporal and frequency complexity
- **Cognitive Relevance**: Comprehensive measure of neural complexity
- **Clinical Application**: Wavelet entropy captures multi-resolution complexity

#### Differential Entropy (`differential_entropy`)
- **Definition**: Entropy of continuous signals
- **Formula**: $H_{diff} = \frac{1}{2}\log(2\pi e \sigma^2)$ where $\sigma^2$ is signal variance
- **Physiological Significance**: Measures the uncertainty in neural signals
- **Cognitive Relevance**: Altered differential entropy may indicate cognitive dysfunction
- **Clinical Application**: Differential entropy changes in cognitive disorders

### 5. Entropy Features

#### Approximate Entropy (`apen`)
- **Definition**: Measure of signal regularity and predictability
- **Formula**: $ApEn(m,r,N) = \phi^m(r) - \phi^{m+1}(r)$ where $\phi^m(r) = \frac{1}{N-m+1}\sum_{i=1}^{N-m+1}\ln C_i^m(r)$
- **Algorithm**: Based on template matching in the signal
- **Physiological Significance**: Quantifies the complexity of neural dynamics
- **Cognitive Relevance**: Reduced ApEn may indicate decreased brain complexity
- **Clinical Application**: ApEn is decreased in Alzheimer's disease

#### Sample Entropy (`sampen`)
- **Definition**: Improved version of approximate entropy
- **Formula**: $SampEn(m,r,N) = -\ln\frac{A}{B}$ where $A$ and $B$ are template matches
- **Advantages**: Less sensitive to data length, more consistent
- **Physiological Significance**: Better measure of signal complexity
- **Cognitive Relevance**: More reliable for cognitive assessment
- **Clinical Application**: Sample entropy is more robust than approximate entropy

#### Permutation Entropy (`pe`)
- **Definition**: Entropy based on ordinal patterns in the signal
- **Formula**: $PE = -\sum_{\pi} p(\pi) \log p(\pi)$ where $\pi$ represents ordinal patterns
- **Algorithm**: Analyzes the sequential relationships in the data
- **Physiological Significance**: Captures temporal dynamics of neural activity
- **Cognitive Relevance**: Effective for detecting cognitive changes
- **Clinical Application**: Permutation entropy captures temporal complexity patterns

### 6. Differential Features

#### First-Order Differences
- **Mean (`diff1_mean`)**: Average of first-order differences
  - **Formula**: $\mu_{diff1} = \frac{1}{N-1}\sum_{i=1}^{N-1} (x_{i+1} - x_i)$
- **Standard Deviation (`diff1_std`)**: Variability of first-order differences
  - **Formula**: $\sigma_{diff1} = \sqrt{\frac{1}{N-2}\sum_{i=1}^{N-1} (x_{i+1} - x_i - \mu_{diff1})^2}$
- **Physiological Significance**: Reflects the rate of change in neural activity
- **Cognitive Relevance**: May indicate processing speed
- **Clinical Application**: Processing speed changes in cognitive impairment

#### Second-Order Differences
- **Mean (`diff2_mean`)**: Average of second-order differences
  - **Formula**: $\mu_{diff2} = \frac{1}{N-2}\sum_{i=1}^{N-2} (x_{i+2} - 2x_{i+1} + x_i)$
- **Standard Deviation (`diff2_std`)**: Variability of second-order differences
  - **Formula**: $\sigma_{diff2} = \sqrt{\frac{1}{N-3}\sum_{i=1}^{N-2} (x_{i+2} - 2x_{i+1} + x_i - \mu_{diff2})^2}$
- **Physiological Significance**: Reflects acceleration in neural activity
- **Cognitive Relevance**: May indicate cognitive flexibility
- **Clinical Application**: Cognitive flexibility assessment

#### Third-Order Differences
- **Mean (`diff3_mean`)**: Average of third-order differences
  - **Formula**: $\mu_{diff3} = \frac{1}{N-3}\sum_{i=1}^{N-3} (x_{i+3} - 3x_{i+2} + 3x_{i+1} - x_i)$
- **Standard Deviation (`diff3_std`)**: Variability of third-order differences
  - **Formula**: $\sigma_{diff3} = \sqrt{\frac{1}{N-4}\sum_{i=1}^{N-3} (x_{i+3} - 3x_{i+2} + 3x_{i+1} - x_i - \mu_{diff3})^2}$
- **Physiological Significance**: Reflects higher-order dynamics
- **Cognitive Relevance**: May indicate complex cognitive processing
- **Clinical Application**: Complex cognitive processing assessment

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
4. **Epoch Length**: 1-4 seconds per epoch depending on cognitive task

### Feature Extraction Pipeline
1. **Segmentation**: Divide continuous EEG into task-specific epochs
2. **Feature Computation**: Extract all 27 features for each epoch
3. **Aggregation**: Compute statistics (mean, std) across epochs within each task
4. **Normalization**: Apply z-score normalization or min-max scaling
5. **Quality Control**: Handle missing data and outliers

### Quality Control Measures
- **Missing Data**: Interpolation or exclusion of epochs with >20% missing data
- **Outlier Detection**: Identify values >3 standard deviations from mean
- **Consistency Checks**: Verify feature ranges and distributions
- **Signal Quality**: Ensure adequate signal-to-noise ratio (>10 dB)
<!-- 
## Clinical Significance and Applications

### Cognitive Assessment Biomarkers
EEG features provide objective measures of brain function that complement traditional cognitive tests:

- **Early Detection**: Detect cognitive changes before behavioral symptoms appear
- **Progression Monitoring**: Track disease progression and treatment response
- **Neural Compensation**: Identify compensatory mechanisms in the brain
- **Treatment Efficacy**: Monitor response to cognitive interventions

### Disease-Specific Applications

#### Alzheimer's Disease
- **Alpha Power Reduction**: Characteristic decrease in 8-12 Hz activity
- **Delta Power Increase**: Increased slow-wave activity in severe cases
- **Complexity Reduction**: Decreased entropy and fractal dimension measures
- **Functional Connectivity**: Altered network connectivity patterns

#### Mild Cognitive Impairment (MCI)
- **Theta Power Changes**: Early indicators of memory dysfunction
- **Beta Activity Alterations**: Changes in cognitive processing
- **Entropy Measures**: Subtle complexity changes before dementia onset
- **Spectral Ratios**: Alpha/theta and beta/theta ratio changes

#### Frontotemporal Dementia
- **Gamma Activity Changes**: Altered high-frequency oscillations
- **Complexity Measures**: Specific patterns in entropy features
- **Differential Features**: Changes in signal dynamics

## Technical Considerations

### Signal Processing Requirements
- **Computational Complexity**: Some features (e.g., Lyapunov exponent) require significant computation
- **Memory Requirements**: Large datasets require efficient processing algorithms
- **Real-time Processing**: Consider computational constraints for clinical applications
- **Robustness**: Features should be robust to artifacts and noise

### Statistical Considerations
- **Multiple Comparisons**: Correct for multiple testing when using many features
- **Effect Sizes**: Consider practical significance beyond statistical significance
- **Longitudinal Analysis**: Account for within-subject correlations
- **Cross-validation**: Ensure robust model performance

## References

### Key Literature

1. **EEG and Cognitive Assessment**
   - Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance. *Brain Research Reviews*, 29(2-3), 169-195.
   - Babiloni, C., et al. (2016). What electrophysiology tells us about Alzheimer's disease: a window into the synchronization and connectivity of brain neurons. *Neurobiology of Aging*, 31(4), 533-548.

2. **Non-linear Analysis**
   - Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. *Proceedings of the National Academy of Sciences*, 88(6), 2297-2301.
   - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology-Heart and Circulatory Physiology*, 278(6), H2039-H2049.
   - Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.

3. **Fractal Analysis**
   - Higuchi, T. (1990). Approach to an irregular time series on the basis of the fractal theory. *Physica D: Nonlinear Phenomena*, 31(2), 277-283.
   - Spasic, S., et al. (2005). Fractal analysis of rat brain activity after injury. *Medical & Biological Engineering & Computing*, 43(3), 345-348.

4. **Complexity Measures**
   - Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory*, 22(1), 75-81.
   - Abásolo, D., et al. (2006). Analysis of EEG background activity in Alzheimer's disease patients with Lempel-Ziv complexity and central tendency measure. *Medical Engineering & Physics*, 28(4), 315-322.

5. **Spectral Analysis**
   - Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70-73.
   - Coifman, R. R., & Wickerhauser, M. V. (1992). Entropy-based algorithms for best basis selection. *IEEE Transactions on Information Theory*, 38(2), 713-718.

6. **Clinical Applications**
   - Jeong, J. (2004). EEG dynamics in patients with Alzheimer's disease. *Clinical Neurophysiology*, 115(7), 1490-1505.
   - Dauwels, J., et al. (2010). Slowing and loss of complexity in Alzheimer's EEG: two sides of the same coin? *International Journal of Alzheimer's Disease*, 2011, 539621. -->

### Open-Source Code Libraries

#### Python Libraries
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
   - Features: Approximate entropy, sample entropy, permutation entropy, spectral entropy
   - Documentation: https://raphaelvallat.com/antropy/

4. **NeuroKit2**: Comprehensive neurophysiological signal processing
   - Repository: https://github.com/neuropsychology/NeuroKit
   - Features: ECG, EEG, EDA, EMG processing and analysis
   - Documentation: https://neurokit2.readthedocs.io/

5. **EEGLAB**: MATLAB toolbox for EEG analysis
   - Website: https://sccn.ucsd.edu/eeglab/
   - Features: ICA, artifact removal, spectral analysis, connectivity
   - Tutorial: https://sccn.ucsd.edu/wiki/EEGLAB


<!-- #### R Libraries
1. **eegkit**: EEG data analysis toolkit
   - Repository: https://github.com/cran/eegkit
   - Features: Spectral analysis, artifact removal, statistical testing

2. **eegUtils**: EEG analysis utilities
   - Repository: https://github.com/craddm/eegUtils
   - Features: Preprocessing, artifact removal, visualization -->

#### MATLAB Toolboxes
1. **FieldTrip**: Advanced EEG/MEG analysis
   - Website: https://www.fieldtriptoolbox.org/
   - Features: Source analysis, connectivity, statistics, real-time processing

2. **SPM**: Statistical Parametric Mapping
   - Website: https://www.fil.ion.ucl.ac.uk/spm/
   - Features: Statistical analysis, source reconstruction, group analysis

### Online Resources and Tutorials
1. **EEG Signal Processing Tutorial**: https://www.mathworks.com/help/signal/examples/analyzing-brain-signals.html
2. **MNE-Python Tutorial**: https://mne.tools/stable/auto_tutorials/index.html
3. **FieldTrip Tutorial**: https://www.fieldtriptoolbox.org/tutorial/
4. **EEGLAB Tutorial**: https://sccn.ucsd.edu/wiki/EEGLAB

## Conclusion

The comprehensive set of 27 EEG features described in this document provides a robust foundation for cognitive impairment assessment. These features capture multiple aspects of neural activity including statistical properties, frequency content, non-linear dynamics, and complexity measures. When combined with appropriate signal processing techniques and clinical validation, these features can serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders.

The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of brain activity.
