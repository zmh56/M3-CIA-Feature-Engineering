# ECG Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of ECG (Electrocardiography) features extracted for cognitive impairment assessment in the M3-CIA framework. These features capture heart rate variability (HRV), cardiac morphology, autonomic nervous system function, and complex dynamic patterns that are closely linked to cognitive performance and brain health.

## Overview

ECG features are extracted from cognitive tasks with 97 features per task, capturing the cardiac activity patterns associated with cognitive processing. These features serve as objective biomarkers for detecting and monitoring cognitive impairment through autonomic nervous system assessment.

## Feature Categories and Descriptions

### 1. Signal Morphology Features

#### Signal Kurtosis (`Signal_Kurtosis`)
- **Definition**: Fourth standardized moment measuring tail heaviness of ECG signal distribution
- **Formula**: $K = \frac{E[(X-\mu)^4]}{\sigma^4} - 3 = \frac{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^4}{\sigma^4} - 3$
- **Physiological Significance**: Indicates the shape of ECG signal distribution and presence of outliers
- **Cognitive Relevance**: May reflect autonomic nervous system stability and cardiac health
- **Clinical Application**: Altered kurtosis may indicate cardiac dysfunction affecting brain perfusion

#### Signal Skewness (`Signal_Skewness`)
- **Definition**: Third standardized moment measuring asymmetry of ECG signal distribution
- **Formula**: $S = \frac{E[(X-\mu)^3]}{\sigma^3} = \frac{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^3}{\sigma^3}$
- **Physiological Significance**: Indicates asymmetry in cardiac electrical activity patterns
- **Cognitive Relevance**: Altered skewness may indicate autonomic dysfunction affecting cognition
- **Clinical Application**: Skewness changes are associated with cardiac arrhythmias and autonomic imbalance

#### Signal Min/Max (`Signal_Min`, `Signal_Max`)
- **Definition**: Minimum and maximum values of ECG signal amplitude
- **Formula**: $Min = \min(x_i)$, $Max = \max(x_i)$
- **Physiological Significance**: Reflects the range of cardiac electrical activity
- **Cognitive Relevance**: May indicate cardiac health status affecting brain function
- **Clinical Application**: Amplitude changes indicate cardiac contractility and electrical conduction

### 2. Time-Domain Heart Rate Variability Features

#### Mean NN Interval (`HRV_MeanNN`)
- **Definition**: Average interval between consecutive normal heartbeats
- **Formula**: $MeanNN = \frac{1}{N}\sum_{i=1}^{N} RR_i$ (in milliseconds)
- **Physiological Significance**: Indicates average heart rate and baseline cardiac function
- **Cognitive Relevance**: Baseline cardiac function affects cognitive performance and brain perfusion
- **Clinical Application**: MeanNN changes indicate overall cardiac health and autonomic tone

#### Standard Deviation of NN Intervals (`HRV_SDNN`)
- **Definition**: Standard deviation of RR intervals, overall HRV measure
- **Formula**: $SDNN = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(RR_i - MeanNN)^2}$
- **Physiological Significance**: Overall heart rate variability reflecting total autonomic activity
- **Cognitive Relevance**: Reduced SDNN may indicate autonomic dysfunction affecting cognition
- **Clinical Application**: SDNN is a strong predictor of cardiovascular and cognitive health

#### Root Mean Square of Successive Differences (`HRV_RMSSD`)
- **Definition**: Square root of mean squared differences between consecutive RR intervals
- **Formula**: $RMSSD = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$
- **Physiological Significance**: Short-term heart rate variability primarily reflecting parasympathetic activity
- **Cognitive Relevance**: Important for cognitive flexibility, attention, and stress recovery
- **Clinical Application**: RMSSD is a key marker of parasympathetic nervous system function

#### Standard Deviation of Successive Differences (`HRV_SDSD`)
- **Definition**: Standard deviation of differences between consecutive RR intervals
- **Formula**: $SDSD = \sqrt{\frac{1}{N-2}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i - \overline{\Delta RR})^2}$
- **Physiological Significance**: Another measure of short-term heart rate variability
- **Cognitive Relevance**: Complements RMSSD for parasympathetic assessment
- **Clinical Application**: SDSD provides additional information about beat-to-beat variability

#### Coefficient of Variation (`HRV_CVNN`, `HRV_CVSD`)
- **Definition**: Relative variability measures
- **Formula**: $CVNN = \frac{SDNN}{MeanNN}$, $CVSD = \frac{RMSSD}{MeanNN}$
- **Physiological Significance**: Normalized measures of heart rate variability
- **Cognitive Relevance**: Relative variability may be more stable across individuals
- **Clinical Application**: Useful for comparing HRV across different heart rate ranges

#### Median NN Interval (`HRV_MedianNN`)
- **Definition**: Median of RR intervals
- **Formula**: $MedianNN = \text{median}(RR_i)$
- **Physiological Significance**: Robust measure of central tendency, less affected by outliers
- **Cognitive Relevance**: May reflect stable cardiac rhythm patterns
- **Clinical Application**: MedianNN is less sensitive to ectopic beats than MeanNN

#### Median Absolute Deviation (`HRV_MadNN`)
- **Definition**: Median absolute deviation of NN intervals
- **Formula**: $MadNN = \text{median}(|RR_i - \text{median}(RR_j)|)$
- **Physiological Significance**: Robust measure of variability, resistant to outliers
- **Cognitive Relevance**: Indicates consistency in heart rate patterns
- **Clinical Application**: MadNN provides robust variability assessment

#### Interquartile Range (`HRV_IQRNN`)
- **Definition**: Difference between 75th and 25th percentiles of RR intervals
- **Formula**: $IQRNN = Q_3 - Q_1$
- **Physiological Significance**: Reflects the dispersion of heart rate distribution
- **Cognitive Relevance**: May indicate range of cardiac responsiveness
- **Clinical Application**: IQRNN indicates the spread of heart rate values

#### Percentile Features (`HRV_Prc20NN`, `HRV_Prc80NN`)
- **Definition**: 20th and 80th percentiles of NN intervals
- **Formula**: $Prc20NN = P_{20}(RR_i)$, $Prc80NN = P_{80}(RR_i)$
- **Physiological Significance**: Indicate the range of heart rate values
- **Cognitive Relevance**: May reflect cardiac response range to cognitive demands
- **Clinical Application**: Percentile measures provide distribution information

#### Percentage of NN50 (`HRV_pNN50`, `HRV_pNN20`)
- **Definition**: Percentage of consecutive RR intervals differing by more than 50ms or 20ms
- **Formula**: $pNN50 = \frac{count(|RR_{i+1} - RR_i| > 50ms)}{N-1} \times 100$
- **Physiological Significance**: Parasympathetic nervous system activity indicator
- **Cognitive Relevance**: Linked to cognitive performance and stress response
- **Clinical Application**: pNN50 is a key parasympathetic function marker

#### Min/Max NN Intervals (`HRV_MinNN`, `HRV_MaxNN`)
- **Definition**: Minimum and maximum RR intervals
- **Formula**: $MinNN = \min(RR_i)$, $MaxNN = \max(RR_i)$
- **Physiological Significance**: Extreme heart rate values indicating cardiac range
- **Cognitive Relevance**: May reflect cardiac adaptability to cognitive demands
- **Clinical Application**: Min/Max NN indicate cardiac reserve and responsiveness

#### Heart Rate Triangle Index (`HRV_HTI`)
- **Definition**: Geometric measure derived from NN interval histogram
- **Formula**: $HTI = \frac{N}{MaxNN - MinNN}$
- **Physiological Significance**: Overall heart rate variability geometric measure
- **Cognitive Relevance**: May reflect overall cardiac health affecting cognition
- **Clinical Application**: HTI provides geometric assessment of HRV

#### Triangular Interpolation of NN Histogram (`HRV_TINN`)
- **Definition**: Width of triangular interpolation of RR interval histogram
- **Formula**: $TINN = \text{width of triangular interpolation}$
- **Physiological Significance**: Geometric measure of heart rate variability
- **Cognitive Relevance**: May reflect overall autonomic balance
- **Clinical Application**: TINN provides histogram-based HRV assessment

#### Ratio Features (`HRV_SDRMSSD`)
- **Definition**: Ratio of SDNN to RMSSD
- **Formula**: $SDRMSSD = \frac{SDNN}{RMSSD}$
- **Physiological Significance**: Balance between overall and short-term variability
- **Cognitive Relevance**: May indicate different aspects of autonomic function
- **Clinical Application**: SDRMSSD reflects the relationship between variability measures

### 3. Frequency-Domain Heart Rate Variability Features

#### Low Frequency Power (`HRV_LF`)
- **Frequency Range**: 0.04-0.15 Hz
- **Formula**: $LF = \sum_{f=0.04}^{0.15} P(f)$ where $P(f)$ is power spectral density
- **Physiological Significance**: Reflects both sympathetic and parasympathetic activity
- **Cognitive Relevance**: Associated with cognitive effort, attention, and mental load
- **Clinical Application**: LF power changes indicate autonomic nervous system activity

#### High Frequency Power (`HRV_HF`)
- **Frequency Range**: 0.15-0.4 Hz
- **Formula**: $HF = \sum_{f=0.15}^{0.4} P(f)$
- **Physiological Significance**: Primarily reflects parasympathetic activity (vagal tone)
- **Cognitive Relevance**: Important for cognitive recovery, relaxation, and attention
- **Clinical Application**: HF power is a key parasympathetic function indicator

#### Very High Frequency Power (`HRV_VHF`)
- **Frequency Range**: 0.4-1.0 Hz
- **Formula**: $VHF = \sum_{f=0.4}^{1.0} P(f)$
- **Physiological Significance**: Physiological significance not fully understood
- **Cognitive Relevance**: May reflect additional autonomic mechanisms
- **Clinical Application**: VHF is often excluded from analysis due to unclear significance

#### Total Power (`HRV_TP`)
- **Definition**: Total power across all frequency ranges
- **Formula**: $TP = \sum_{f} P(f)$
- **Physiological Significance**: Overall heart rate variability across all frequencies
- **Cognitive Relevance**: Indicates total autonomic nervous system activity
- **Clinical Application**: TP reflects overall cardiovascular health

#### LF/HF Ratio (`HRV_LFHF`)
- **Definition**: Ratio of low frequency to high frequency power
- **Formula**: $LF/HF = \frac{LF}{HF}$
- **Physiological Significance**: Sympathovagal balance indicator
- **Cognitive Relevance**: Indicator of stress, cognitive load, and autonomic balance
- **Clinical Application**: LF/HF ratio is a key autonomic balance measure

#### Normalized LF Power (`HRV_LFn`)
- **Definition**: LF power normalized by total power minus VHF
- **Formula**: $LF_n = \frac{LF}{TP - VHF}$
- **Physiological Significance**: Relative sympathetic activity measure
- **Cognitive Relevance**: May indicate cognitive stress levels and effort
- **Clinical Application**: Normalized measures reduce individual differences

#### Normalized HF Power (`HRV_HFn`)
- **Definition**: HF power normalized by total power minus VHF
- **Formula**: $HF_n = \frac{HF}{TP - VHF}$
- **Physiological Significance**: Relative parasympathetic activity measure
- **Cognitive Relevance**: Associated with cognitive recovery and attention
- **Clinical Application**: Normalized HF provides relative parasympathetic assessment

#### Natural Logarithm of HF (`HRV_LnHF`)
- **Definition**: Natural logarithm of high-frequency power
- **Formula**: $LnHF = \ln(HF)$
- **Physiological Significance**: Used to reduce data skewness and improve normality
- **Cognitive Relevance**: Provides normalized measure for statistical analysis
- **Clinical Application**: Log transformation improves statistical properties

### 4. Non-Linear Heart Rate Variability Features

#### Poincaré Plot Features

##### SD1 (`HRV_SD1`)
- **Definition**: Standard deviation perpendicular to the line of identity in Poincaré plot
- **Formula**: $SD1 = \frac{SDNN}{\sqrt{2}} \cdot \sqrt{1 - \rho}$ where $\rho$ is correlation
- **Physiological Significance**: Short-term heart rate variability (parasympathetic activity)
- **Cognitive Relevance**: Parasympathetic activity affecting cognitive performance
- **Clinical Application**: SD1 is a key short-term variability measure

##### SD2 (`HRV_SD2`)
- **Definition**: Standard deviation along the line of identity in Poincaré plot
- **Formula**: $SD2 = \sqrt{2 \cdot SDNN^2 - SD1^2}$
- **Physiological Significance**: Long-term heart rate variability (sympathetic activity)
- **Cognitive Relevance**: Overall autonomic function affecting cognitive processes
- **Clinical Application**: SD2 reflects long-term autonomic regulation

##### SD1/SD2 Ratio (`HRV_SD1SD2`)
- **Definition**: Ratio of SD1 to SD2
- **Formula**: $SD1/SD2 = \frac{SD1}{SD2}$
- **Physiological Significance**: Balance between short and long-term variability
- **Cognitive Relevance**: May indicate autonomic balance affecting cognition
- **Clinical Application**: SD1/SD2 ratio indicates autonomic balance

##### Poincaré Plot Area (`HRV_S`)
- **Definition**: Area of the Poincaré plot
- **Formula**: $S = \pi \cdot SD1 \cdot SD2$
- **Physiological Significance**: Overall dynamics of heart rate variability
- **Cognitive Relevance**: May reflect overall cardiac health affecting brain function
- **Clinical Application**: Poincaré area indicates overall HRV dynamics

#### Complexity Indices

##### Complexity Index (`HRV_CSI`)
- **Definition**: SD2/SD1 ratio, complexity measure
- **Formula**: $CSI = \frac{SD2}{SD1}$
- **Physiological Significance**: Complexity of heart rate variability patterns
- **Cognitive Relevance**: May indicate cognitive complexity and adaptability
- **Clinical Application**: CSI reflects heart rate pattern complexity

##### Complexity Variability Index (`HRV_CVI`)
- **Definition**: Logarithmic product of SD1 and SD2
- **Formula**: $CVI = \log(SD1 \cdot SD2)$
- **Physiological Significance**: Complexity and irregularity of heart rate sequence
- **Cognitive Relevance**: May indicate cognitive flexibility and adaptability
- **Clinical Application**: CVI provides logarithmic complexity measure

##### Modified Complexity Index (`HRV_CSI_Modified`)
- **Definition**: Modified complexity index with different calculation approach
- **Physiological Significance**: Alternative complexity assessment method
- **Cognitive Relevance**: May provide additional complexity information
- **Clinical Application**: Modified CSI offers alternative complexity measure

#### Fractal Analysis Features

##### Detrended Fluctuation Analysis Alpha 2 (`HRV_DFA_alpha2`)
- **Definition**: Alpha2 coefficient in fractal dimension analysis
- **Formula**: $DFA_{\alpha2} = \frac{\log(F(n))}{\log(n)}$ for long-term correlations
- **Physiological Significance**: Self-similarity of long-term heart rate variability
- **Cognitive Relevance**: May indicate cognitive complexity and long-term patterns
- **Clinical Application**: DFA alpha2 reflects long-range correlations

##### Multifractal Detrended Fluctuation Analysis Features

###### MFDFA Alpha 2 Width (`HRV_MFDFA_alpha2_Width`)
- **Definition**: Width of the multifractal spectrum
- **Formula**: $Width = \alpha_{max} - \alpha_{min}$
- **Physiological Significance**: Multifractal characteristics of heart rate signals
- **Cognitive Relevance**: Indicates complexity of autonomic regulation
- **Clinical Application**: MFDFA width reflects multifractal diversity

###### MFDFA Alpha 2 Peak (`HRV_MFDFA_alpha2_Peak`)
- **Definition**: Peak value in the multifractal spectrum
- **Formula**: $Peak = f(\alpha_{peak})$
- **Physiological Significance**: Strongest multifractal characteristic of heart rate signal
- **Cognitive Relevance**: May reflect dominant cognitive processing patterns
- **Clinical Application**: MFDFA peak indicates primary multifractal behavior

###### MFDFA Alpha 2 Mean (`HRV_MFDFA_alpha2_Mean`)
- **Definition**: Mean value in the multifractal spectrum
- **Formula**: $Mean = \frac{1}{n}\sum_{i=1}^{n} f(\alpha_i)$
- **Physiological Significance**: Overall multifractal characteristics
- **Cognitive Relevance**: May indicate average cognitive complexity
- **Clinical Application**: MFDFA mean reflects overall multifractal properties

###### MFDFA Alpha 2 Max (`HRV_MFDFA_alpha2_Max`)
- **Definition**: Maximum value in the multifractal spectrum
- **Formula**: $Max = \max(f(\alpha_i))$
- **Physiological Significance**: Maximum multifractal characteristic
- **Cognitive Relevance**: May indicate peak cognitive complexity
- **Clinical Application**: MFDFA max indicates maximum multifractal strength

###### MFDFA Alpha 2 Delta (`HRV_MFDFA_alpha2_Delta`)
- **Definition**: Delta value in the multifractal spectrum
- **Formula**: $Delta = \alpha_{max} - \alpha_{min}$
- **Physiological Significance**: Diversity of heart rate signals
- **Cognitive Relevance**: May indicate cognitive diversity and flexibility
- **Clinical Application**: MFDFA delta reflects multifractal range

###### MFDFA Alpha 2 Asymmetry (`HRV_MFDFA_alpha2_Asymmetry`)
- **Definition**: Asymmetry in the multifractal spectrum
- **Formula**: $Asymmetry = \frac{|\alpha_{left} - \alpha_{right}|}{|\alpha_{left} + \alpha_{right}|}$
- **Physiological Significance**: Directional behavior of heart rate variability
- **Cognitive Relevance**: May indicate pathological autonomic patterns
- **Clinical Application**: MFDFA asymmetry reflects directional multifractal behavior

###### MFDFA Alpha 2 Fluctuation (`HRV_MFDFA_alpha2_Fluctuation`)
- **Definition**: Fluctuation in the multifractal spectrum
- **Formula**: $Fluctuation = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(f(\alpha_i) - \overline{f(\alpha)})^2}$
- **Physiological Significance**: Stability of heart rate signals
- **Cognitive Relevance**: May indicate cognitive stability
- **Clinical Application**: MFDFA fluctuation reflects multifractal stability

###### MFDFA Alpha 2 Increment (`HRV_MFDFA_alpha2_Increment`)
- **Definition**: Increment in the multifractal spectrum
- **Formula**: $Increment = \sum_{i=1}^{n-1}|f(\alpha_{i+1}) - f(\alpha_i)|$
- **Physiological Significance**: Trend of heart rate variability
- **Cognitive Relevance**: May indicate cognitive trend patterns
- **Clinical Application**: MFDFA increment reflects multifractal trends

### 5. Entropy-Based Features

#### Approximate Entropy (`HRV_ApEn`)
- **Definition**: Measure of regularity in RR interval sequence
- **Formula**: $ApEn(m,r,N) = \phi^m(r) - \phi^{m+1}(r)$ where $\phi^m(r) = \frac{1}{N-m+1}\sum_{i=1}^{N-m+1}\ln C_i^m(r)$
- **Physiological Significance**: Quantifies complexity of heart rate patterns
- **Cognitive Relevance**: Reduced ApEn may indicate decreased autonomic complexity
- **Clinical Application**: ApEn is decreased in various pathological conditions

#### Sample Entropy (`HRV_sampEn`)
- **Definition**: Improved measure of signal complexity
- **Formula**: $SampEn(m,r,N) = -\ln\frac{A}{B}$ where $A$ and $B$ are template matches
- **Physiological Significance**: Better measure of heart rate regularity
- **Cognitive Relevance**: More reliable for cognitive assessment
- **Clinical Application**: Sample entropy is more robust than approximate entropy

#### Shannon Entropy (`HRV_shanEn`)
- **Definition**: Information content of RR interval distribution
- **Formula**: $H = -\sum_{i} p_i \log p_i$ where $p_i$ is probability of interval $i$
- **Physiological Significance**: Measures uncertainty in heart rate patterns
- **Cognitive Relevance**: May indicate cognitive load and stress
- **Clinical Application**: Shannon entropy reflects information content

#### Fuzzy Entropy (`HRV_FuzzyEn`)
- **Definition**: Entropy measure using fuzzy logic
- **Formula**: $FuzzyEn(m,r,N) = -\ln\frac{\Phi^m(r)}{\Phi^{m+1}(r)}$
- **Physiological Significance**: Improved entropy calculation with fuzzy logic
- **Cognitive Relevance**: Better complexity assessment for cognitive applications
- **Clinical Application**: Fuzzy entropy provides robust complexity measures

#### Multi-Scale Entropy (`HRV_MSEn`)
- **Definition**: Entropy analysis across different time scales
- **Formula**: $MSEn(\tau) = SampEn(\tau)$ for scale factor $\tau$
- **Physiological Significance**: Analyzes complexity across different time scales
- **Cognitive Relevance**: May indicate multi-scale cognitive complexity
- **Clinical Application**: MSEn provides scale-dependent complexity assessment

#### Composite Multi-Scale Entropy (`HRV_CMSEn`)
- **Definition**: Improvement to multi-scale entropy
- **Formula**: $CMSEn(\tau) = -\ln\frac{\langle A(\tau) \rangle}{\langle B(\tau) \rangle}$
- **Physiological Significance**: Increased sensitivity to heart rate variability analysis
- **Cognitive Relevance**: More sensitive cognitive complexity assessment
- **Clinical Application**: CMSEn provides enhanced multi-scale analysis

#### Refined Composite Multi-Scale Entropy (`HRV_RCMSEn`)
- **Definition**: Further improvement to multi-scale entropy calculation
- **Formula**: $RCMSEn(\tau) = -\ln\frac{\sum_{j=1}^{\tau}A_j(\tau)}{\sum_{j=1}^{\tau}B_j(\tau)}$
- **Physiological Significance**: Most refined multi-scale entropy measure
- **Cognitive Relevance**: Most accurate cognitive complexity assessment
- **Clinical Application**: RCMSEn provides the most robust multi-scale analysis

### 6. Heart Rate Asymmetry Features (HRA)

#### Guzik Index (`HRV_GI`)
- **Definition**: Asymmetry of NN intervals in Poincaré plot between acceleration and deceleration
- **Formula**: $GI = \frac{4\sum_{i=1}^{N-1}(RR_i \cdot RR_{i+1})}{(RR_1^2 + RR_N^2 + 2\sum_{i=1}^{N-1}RR_i^2)}$
- **Physiological Significance**: Asymmetry between heart rate acceleration and deceleration
- **Cognitive Relevance**: May indicate cognitive processing asymmetry
- **Clinical Application**: GI reflects autonomic asymmetry

#### Slope Index (`HRV_SI`)
- **Definition**: Degree of slope in Poincaré plot shape
- **Formula**: $SI = \frac{\sum_{i=1}^{N-1}(RR_i - \overline{RR})(RR_{i+1} - \overline{RR})}{\sum_{i=1}^{N-1}(RR_i - \overline{RR})^2}$
- **Physiological Significance**: Asymmetry in heart rate variability patterns
- **Cognitive Relevance**: May indicate cognitive response asymmetry
- **Clinical Application**: SI reflects directional heart rate patterns

#### Area Index (`HRV_AI`)
- **Definition**: Asymmetry in heart rate variability by calculating area of specific regions
- **Formula**: $AI = \frac{Area_{above} - Area_{below}}{Area_{above} + Area_{below}}$
- **Physiological Significance**: Asymmetry evaluation in Poincaré plot regions
- **Cognitive Relevance**: May indicate cognitive processing asymmetry
- **Clinical Application**: AI provides geometric asymmetry assessment

#### Porta Index (`HRV_PI`)
- **Definition**: Ratio of increasing to decreasing RR intervals
- **Formula**: $PI = \frac{count(RR_{i+1} > RR_i)}{count(RR_{i+1} < RR_i)}$
- **Physiological Significance**: Asymmetry in heart rate variability direction
- **Cognitive Relevance**: May indicate cognitive response directionality
- **Clinical Application**: PI reflects directional heart rate asymmetry

#### Acceleration/Deceleration Contributions

##### Short-term Contributions (`HRV_C1d`, `HRV_C1a`)
- **Definition**: Contribution of deceleration/acceleration to short-term variability
- **Formula**: $C1d = \frac{SD1_d}{SD1}$, $C1a = \frac{SD1_a}{SD1}$
- **Physiological Significance**: Sympathetic vs parasympathetic contributions
- **Cognitive Relevance**: May indicate cognitive stress vs recovery patterns
- **Clinical Application**: C1d/C1a reflect autonomic balance components

##### Short-term Standard Deviations (`HRV_SD1d`, `HRV_SD1a`)
- **Definition**: Short-term standard deviation of deceleration/acceleration
- **Formula**: $SD1d = SD1 \cdot C1d$, $SD1a = SD1 \cdot C1a$
- **Physiological Significance**: Separate short-term variability components
- **Cognitive Relevance**: May indicate separate cognitive processing components
- **Clinical Application**: SD1d/SD1a provide component-specific variability

##### Long-term Contributions (`HRV_C2d`, `HRV_C2a`)
- **Definition**: Contribution of deceleration/acceleration to long-term variability
- **Formula**: $C2d = \frac{SD2_d}{SD2}$, $C2a = \frac{SD2_a}{SD2}$
- **Physiological Significance**: Long-term autonomic contributions
- **Cognitive Relevance**: May indicate long-term cognitive patterns
- **Clinical Application**: C2d/C2a reflect long-term autonomic components

##### Long-term Standard Deviations (`HRV_SD2d`, `HRV_SD2a`)
- **Definition**: Long-term standard deviation of deceleration/acceleration
- **Formula**: $SD2d = SD2 \cdot C2d$, $SD2a = SD2 \cdot C2a$
- **Physiological Significance**: Separate long-term variability components
- **Cognitive Relevance**: May indicate separate long-term cognitive patterns
- **Clinical Application**: SD2d/SD2a provide long-term component variability

##### Total Contributions (`HRV_Cd`, `HRV_Ca`)
- **Definition**: Total contribution of deceleration/acceleration to variability
- **Formula**: $Cd = \frac{SDNN_d}{SDNN}$, $Ca = \frac{SDNN_a}{SDNN}$
- **Physiological Significance**: Total autonomic contributions
- **Cognitive Relevance**: May indicate total cognitive processing components
- **Clinical Application**: Cd/Ca reflect total autonomic balance

##### Total Standard Deviations (`HRV_SDNNd`, `HRV_SDNNa`)
- **Definition**: Total standard deviation of deceleration/acceleration
- **Formula**: $SDNNd = SDNN \cdot Cd$, $SDNNa = SDNN \cdot Ca$
- **Physiological Significance**: Total variability components
- **Cognitive Relevance**: May indicate total cognitive variability components
- **Clinical Application**: SDNNd/SDNNa provide total component variability

### 7. Heart Rate Fragmentation Features (HRF)

#### Percentage of Inflection Points (`HRV_PIP`)
- **Definition**: Percentage of inflection points in NN interval sequence
- **Formula**: $PIP = \frac{count(\text{inflection points})}{N-2} \times 100$
- **Physiological Significance**: Indicates fragmentation in heart rate patterns
- **Cognitive Relevance**: May indicate autonomic dysfunction affecting cognition
- **Clinical Application**: PIP reflects heart rate pattern fragmentation

#### Inverse of Average Length of Segments (`HRV_IALS`)
- **Definition**: Measure of segment length in NN interval sequence
- **Formula**: $IALS = \frac{1}{\overline{L}}$ where $\overline{L}$ is average segment length
- **Physiological Significance**: Indicates continuity of heart rate patterns
- **Cognitive Relevance**: May reflect cognitive stability and continuity
- **Clinical Application**: IALS reflects heart rate pattern continuity

#### Percentage of Short Segments (`HRV_PSS`)
- **Definition**: Percentage of short segments in NN interval sequence
- **Formula**: $PSS = \frac{count(\text{short segments})}{total \text{ segments}} \times 100$
- **Physiological Significance**: Proportion of transient patterns
- **Cognitive Relevance**: May indicate transient cognitive processing
- **Clinical Application**: PSS reflects transient heart rate patterns

#### Percentage of Alternating Segments (`HRV_PAS`)
- **Definition**: Percentage of alternating acceleration/deceleration segments
- **Formula**: $PAS = \frac{count(\text{alternating segments})}{total \text{ segments}} \times 100$
- **Physiological Significance**: Proportion of alternating patterns
- **Cognitive Relevance**: May indicate autonomic rhythm affecting cognition
- **Clinical Application**: PAS reflects alternating heart rate patterns

### 8. Fractal and Complexity Features

#### Correlation Dimension (`HRV_CD`)
- **Definition**: Fractal dimension reflecting correlation structure
- **Formula**: $CD = \lim_{r \to 0} \frac{\log C(r)}{\log r}$ where $C(r)$ is correlation integral
- **Physiological Significance**: Fractal characteristics of heart rate time series
- **Cognitive Relevance**: May indicate cognitive complexity and structure
- **Clinical Application**: CD reflects heart rate correlation structure

#### Higuchi Fractal Dimension (`HRV_HFD`)
- **Definition**: Fractal dimension using Higuchi method
- **Formula**: $HFD = \frac{\log(L(k))}{\log(k)}$ where $L(k)$ is average length
- **Physiological Significance**: Complexity and self-similarity of heart rate time series
- **Cognitive Relevance**: May indicate cognitive complexity and self-similarity
- **Clinical Application**: HFD provides robust fractal dimension measure

#### Katz Fractal Dimension (`HRV_KFD`)
- **Definition**: Fractal dimension using Katz method
- **Formula**: $KFD = \frac{\log(N)}{\log(\frac{d}{L})}$ where $d$ is distance, $L$ is total length
- **Physiological Significance**: Complexity of heart rate time series
- **Cognitive Relevance**: May indicate overall cognitive complexity
- **Clinical Application**: KFD provides alternative fractal dimension measure

#### Lempel-Ziv Complexity (`HRV_LZC`)
- **Definition**: Complexity based on pattern repetition
- **Formula**: $LZC = \frac{c(n)}{n/\log_2(n)}$ where $c(n)$ is number of distinct patterns
- **Physiological Significance**: Complexity and pattern diversity in heart rate time series
- **Cognitive Relevance**: May indicate cognitive pattern complexity
- **Clinical Application**: LZC reflects heart rate pattern complexity

## Implementation Pipeline

### Data Preprocessing
1. **R-Peak Detection**: 
   - Pan-Tompkins algorithm
   - Hamilton algorithm
   - Wavelet-based detection
2. **Artifact Removal**: 
   - Ectopic beat detection and correction
   - Outlier removal (>3 standard deviations)
   - Interpolation of missing beats
3. **Quality Control**: 
   - Minimum data length (>5 minutes)
   - Artifact rate (<5% of beats)
   - Stationarity checks

### Feature Extraction Pipeline
1. **RR Interval Extraction**: Detect R-peaks and compute intervals
2. **Artifact Correction**: Remove outliers and artifacts
3. **Feature Computation**: Calculate all 97 HRV features
4. **Validation**: Check feature ranges and consistency
5. **Normalization**: Apply appropriate scaling for each feature type

### Quality Control Measures
- **Signal Quality**: Ensure adequate signal-to-noise ratio (>10 dB)
- **Missing Data**: Handle gaps with interpolation or exclusion
- **Outlier Detection**: Identify values >3 standard deviations from mean
- **Consistency Checks**: Verify feature ranges and distributions
- **Temporal Stability**: Check for stationarity in the signal
<!-- 
## Clinical Significance and Applications

### Autonomic Function Assessment
ECG features provide comprehensive assessment of autonomic nervous system function:

- **Parasympathetic Activity**: Important for cognitive recovery and attention
- **Sympathetic Activity**: Related to cognitive effort and stress response
- **Autonomic Balance**: Critical for optimal cognitive performance
- **Complexity Measures**: Indicate system adaptability and health

### Cognitive Health Indicators
ECG features can serve as biomarkers for:

- **Cognitive Load**: Heart rate variability changes with mental effort
- **Stress Response**: Autonomic patterns reflect cognitive stress
- **Cognitive Reserve**: HRV may indicate brain health and resilience
- **Attention and Focus**: Parasympathetic activity affects attention
- **Memory Processing**: Autonomic patterns during memory tasks

### Disease-Specific Applications

#### Alzheimer's Disease
- **Reduced HRV**: Decreased overall heart rate variability
- **Parasympathetic Decline**: Reduced high-frequency power
- **Complexity Loss**: Decreased entropy and fractal measures
- **Autonomic Dysfunction**: Altered sympathovagal balance

#### Mild Cognitive Impairment (MCI)
- **Early HRV Changes**: Subtle autonomic dysfunction
- **Stress Sensitivity**: Altered stress response patterns
- **Complexity Reduction**: Decreased multi-scale entropy
- **Fragmentation**: Increased heart rate fragmentation

#### Cardiovascular Disease
- **HRV Reduction**: Decreased overall variability
- **Sympathetic Overactivity**: Increased LF/HF ratio
- **Complexity Loss**: Reduced fractal and entropy measures
- **Fragmentation**: Increased PIP and IALS

### Early Detection and Monitoring
ECG features may detect cognitive changes before behavioral symptoms:

- **Autonomic Dysfunction**: Often precedes cognitive decline
- **Stress Vulnerability**: May indicate cognitive risk
- **Treatment Response**: Can monitor intervention effects
- **Progression Tracking**: Longitudinal HRV changes

## Technical Considerations

### Signal Processing Requirements
- **Computational Complexity**: Some features (e.g., MFDFA) require significant computation
- **Memory Requirements**: Large datasets require efficient processing algorithms
- **Real-time Processing**: Consider computational constraints for clinical applications
- **Robustness**: Features should be robust to artifacts and noise

### Statistical Considerations
- **Multiple Comparisons**: Correct for multiple testing when using many features
- **Effect Sizes**: Consider practical significance beyond statistical significance
- **Longitudinal Analysis**: Account for within-subject correlations
- **Cross-validation**: Ensure robust model performance
- **Normalization**: Consider individual differences in baseline HRV

### Clinical Validation
- **Population Norms**: Establish normal ranges for different populations
- **Age and Gender Effects**: Account for demographic factors
- **Medication Effects**: Consider drug influences on HRV
- **Environmental Factors**: Account for time of day, activity level
- **Comorbidities**: Consider multiple health conditions

## References

### Key Literature

1. **HRV Standards and Guidelines**
   - Malik, M., et al. (1996). Heart rate variability: Standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
   - Task Force of the European Society of Cardiology. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *European Heart Journal*, 17(3), 354-381.
   - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258.

2. **HRV and Cognitive Function**
   - Thayer, J. F., & Lane, R. D. (2009). Claude Bernard and the heart–brain connection: Further elaboration of a model of neurovisceral integration. *Neuroscience & Biobehavioral Reviews*, 33(2), 81-88.
   - Hansen, A. L., et al. (2003). Heart rate variability and its relation to cognitive function in elderly persons. *Age and Ageing*, 32(6), 601-607.
   - Thayer, J. F., et al. (2012). A meta-analysis of heart rate variability and neuroimaging studies: Implications for heart rate variability as a marker of stress and health. *Neuroscience & Biobehavioral Reviews*, 36(2), 747-756.

3. **Non-linear Analysis**
   - Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. *Proceedings of the National Academy of Sciences*, 88(6), 2297-2301.
   - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology-Heart and Circulatory Physiology*, 278(6), H2039-H2049.
   - Costa, M., et al. (2002). Multiscale entropy analysis of complex physiologic time series. *Physical Review Letters*, 89(6), 068102.

4. **Fractal Analysis**
   - Peng, C. K., et al. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos*, 5(1), 82-87.
   - Ihlen, E. A. F. (2012). Introduction to multifractal detrended fluctuation analysis in Matlab. *Frontiers in Physiology*, 3, 141.
   - Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A: Statistical Mechanics and its Applications*, 316(1-4), 87-114.

5. **Heart Rate Asymmetry**
   - Guzik, P., et al. (2006). Heart rate asymmetry by Poincaré plots of RR intervals. *Biomedical Signal Processing and Control*, 1(1), 33-38.
   - Porta, A., et al. (2008). Heart rate asymmetry in healthy subjects. *Journal of Electrocardiology*, 41(3), 173-177.
   - Karmakar, C. K., et al. (2011). Complex correlation measure: A novel descriptor for Poincaré plot. *Biomedical Engineering Online*, 10(1), 17.

6. **Heart Rate Fragmentation**
   - Costa, M. D., et al. (2017). Heart rate fragmentation: A new approach to the analysis of cardiac interbeat interval dynamics. *Frontiers in Physiology*, 8, 255.
   - Costa, M. D., et al. (2019). Heart rate fragmentation as a novel biomarker of adverse cardiovascular events: The Multi-Ethnic Study of Atherosclerosis. *Psychophysiology*, 56(5), e13319.

7. **Clinical Applications**
   - Thayer, J. F., et al. (2010). A meta-analysis of heart rate variability and neuroimaging studies: Implications for heart rate variability as a marker of stress and health. *Neuroscience & Biobehavioral Reviews*, 36(2), 747-756.
   - Kemp, A. H., et al. (2014). Impact of depression and antidepressant treatment on heart rate variability: A review and meta-analysis. *Biological Psychiatry*, 67(11), 1067-1074.
   - Alvares, G. A., et al. (2016). Autonomic nervous system dysfunction in psychiatric disorders and the impact of psychotropic medications: A systematic review and meta-analysis. *Journal of Psychiatry & Neuroscience*, 41(2), 89-104. -->

### Open-Source Code Libraries

#### Python Libraries
1. **HeartPy**: Heart rate variability analysis
   - Repository: https://github.com/paulvangentcom/heartrate_analysis_python
   - Features: R-peak detection, HRV analysis, artifact removal
   - Documentation: https://python-heart-rate-analysis-toolkit.readthedocs.io/

2. **PyHRV**: Comprehensive HRV analysis
   - Repository: https://github.com/PGomes92/pyhrv
   - Features: Time-domain, frequency-domain, non-linear HRV features
   - Documentation: https://pyhrv.readthedocs.io/

3. **hrv-analysis**: HRV feature extraction and analysis
   - Repository: https://github.com/Aura-healthcare/hrv-analysis
   - Features: Statistical, spectral, and non-linear HRV measures
   - Documentation: https://hrv-analysis.readthedocs.io/

4. **NeuroKit2**: Comprehensive neurophysiological signal processing
   - Repository: https://github.com/neuropsychology/NeuroKit
   - Features: ECG, HRV, EDA, EMG processing and analysis
   - Documentation: https://neurokit2.readthedocs.io/

5. **Antropy**: Entropy measures for time series
   - Repository: https://github.com/raphaelvallat/antropy
   - Features: Approximate entropy, sample entropy, permutation entropy, spectral entropy
   - Documentation: https://raphaelvallat.com/antropy/
<!-- 
#### R Libraries
1. **RHRV**: Heart rate variability analysis
   - Repository: https://github.com/cran/RHRV
   - Features: Comprehensive HRV analysis, non-linear measures
   - Documentation: https://cran.r-project.org/web/packages/RHRV/

2. **HRV**: Heart rate variability analysis
   - Repository: https://github.com/cran/HRV
   - Features: Time-domain and frequency-domain HRV analysis
   - Documentation: https://cran.r-project.org/web/packages/HRV/

#### MATLAB Toolboxes
1. **HRVAS**: Heart rate variability analysis software
   - Website: https://github.com/jramshur/HRVAS
   - Features: Comprehensive HRV analysis, artifact removal
   - Documentation: https://github.com/jramshur/HRVAS

2. **PhysioNet Cardiovascular Signal Toolbox**
   - Website: https://physionet.org/content/cvst/1.0/
   - Features: Cardiovascular signal processing and analysis
   - Documentation: https://physionet.org/content/cvst/1.0/ -->

#### Online Resources and Tutorials
1. **HRV Analysis Tutorial**: https://github.com/Aura-healthcare/hrv-analysis
2. **HeartPy Tutorial**: https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/
3. **PyHRV Documentation**: https://pyhrv.readthedocs.io/
4. **NeuroKit2 ECG Tutorial**: https://neurokit2.readthedocs.io/en/latest/tutorials/ecg.html

### GitHub ECG Feature Extraction Libraries
1. **ECG Feature Extraction using Python**: https://github.com/chandanacharya1/ECG-Feature-extraction-using-Python
2. **Time-Series Feature Extraction ECG**: https://github.com/tkhan11/Time-Series-Feature-Extraction-ECG
3. **ECG Features**: https://github.com/Seb-Good/ecg-features
4. **ECG Analysis**: https://github.com/ecg-analyzer/ecg-analyzer
5. **Heart Rate Variability Analysis**: https://github.com/robertseo/heart-rate-variability

## Conclusion

The comprehensive set of 97 ECG features described in this document provides a robust foundation for cognitive impairment assessment through autonomic nervous system evaluation. These features capture multiple aspects of cardiac activity including time-domain variability, frequency-domain patterns, non-linear dynamics, entropy measures, heart rate asymmetry, fragmentation, and fractal complexity.

When combined with appropriate signal processing techniques and clinical validation, these features can serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders. The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of autonomic nervous system activity.

The extensive literature support and open-source code libraries provide researchers and clinicians with the tools necessary to implement and validate these features in their own studies, contributing to the advancement of cognitive health assessment through cardiovascular monitoring.
