# ECG Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of ECG (Electrocardiography) features extracted for cognitive impairment assessment in the M3-CIA framework. These features capture heart rate variability (HRV), cardiac morphology, autonomic nervous system function, cardiorespiratory coupling, and complex dynamic patterns that are closely linked to cognitive performance and brain health.

## Overview

ECG features are extracted from cognitive tasks, capturing cardiac activity patterns associated with cognitive processing. The feature set spans six major categories: (1) morphological features from P-QRS-T waveform, (2) time-domain HRV, (3) frequency-domain HRV, (4) nonlinear HRV, (5) cardiorespiratory coupling, and (6) entropy/fractal complexity. These features serve as objective biomarkers for detecting and monitoring cognitive impairment through autonomic nervous system and neurocardiac regulation assessment.

## Feature Categories and Descriptions

### 1. Morphological Features (ECG Waveform)

Morphological features quantify the shape and timing of the P-QRS-T complex, reflecting atrioventricular conduction, ventricular depolarization/repolarization, and neurocardiac regulation linked to cognitive function.

#### PR Interval (`PR_Interval_Mean`, `PR_Interval_Std`)
- **Definition**: Interval from end of P wave to start of QRS complex (atrioventricular conduction time)
- **Formula**: $PR = t_{QRS\_start} - t_{P\_end}$ (in ms); Mean and Std over all beats
- **Physiological Significance**: Reflects atrioventricular conduction and neurocardiac regulation
- **Relevance to Cognitive Impairment**: Altered PR interval may indicate autonomic dysfunction affecting brain–heart axis; linked to cognitive decline in elderly populations

#### QRS Duration (`QRS_Duration`)
- **Definition**: Duration of QRS complex (ventricular depolarization)
- **Formula**: $QRS = t_{QRS\_end} - t_{QRS\_start}$ (typically 80–120 ms)
- **Physiological Significance**: Indicates ventricular conduction integrity with autonomic influence
- **Relevance to Cognitive Impairment**: Prolonged QRS may reflect cardiac conduction abnormalities affecting cerebral perfusion and cognitive reserve

#### QT/JT Interval (`QT_Interval`, `JT_Interval`)
- **Definition**: QT = ventricular repolarization duration (QRS onset to T wave end); JT = JT = QT − QRS
- **Formula**: $QT = t_{T\_end} - t_{QRS\_start}$; $JT = QT - QRS$
- **Physiological Significance**: Provides markers of cardiac repolarization relevant to autonomic tone
- **Relevance to Cognitive Impairment**: QT variability and prolongation are associated with autonomic dysfunction and cognitive impairment; repolarization abnormalities may precede cognitive decline

#### P/T Wave Amplitude (`P_Wave_Amplitude`, `T_Wave_Amplitude`)
- **Definition**: Peak amplitudes of atrial depolarization (P wave) and ventricular repolarization (T wave)
- **Formula**: $A_P = \max(P_{wave})$, $A_T = \max(T_{wave})$ relative to baseline
- **Physiological Significance**: Represents cardiac electrical activity strength with systemic and autonomic links
- **Relevance to Cognitive Impairment**: Amplitude changes may reflect neurocardiac coupling alterations in cognitive disorders

#### ST Segment Level (`ST_Segment_Level`)
- **Definition**: Displacement of ST segment from baseline (typically at J-point + 60 ms)
- **Formula**: $ST = V_{ST} - V_{baseline}$ (in mV)
- **Physiological Significance**: Sensitive to autonomic tone changes and physiological stress
- **Relevance to Cognitive Impairment**: ST segment variability may indicate stress-related autonomic dysregulation affecting cognitive load and performance

#### Signal Skewness (`Signal_Skewness`)
- **Definition**: Third standardized moment measuring asymmetry of ECG signal distribution
- **Formula**: $S = \frac{E[(X-\mu)^3]}{\sigma^3} = \frac{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^3}{\sigma^3}$
- **Physiological Significance**: Index of waveform asymmetry; reflects morphology changes related to neurocardiac coupling
- **Relevance to Cognitive Impairment**: Altered skewness may indicate autonomic dysfunction and cardiac arrhythmias affecting cognition

#### Signal Kurtosis (`Signal_Kurtosis`)
- **Definition**: Fourth standardized moment measuring tail heaviness of ECG signal distribution
- **Formula**: $K = \frac{E[(X-\mu)^4]}{\sigma^4} - 3 = \frac{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^4}{\sigma^4} - 3$
- **Physiological Significance**: Indicates the shape of ECG signal distribution and presence of outliers
- **Relevance to Cognitive Impairment**: Altered kurtosis may indicate cardiac dysfunction affecting brain perfusion and autonomic stability

#### Signal Min/Max (`Signal_Min`, `Signal_Max`)
- **Definition**: Minimum and maximum values of ECG signal amplitude
- **Formula**: $Min = \min(x_i)$, $Max = \max(x_i)$
- **Physiological Significance**: Reflects the range of cardiac electrical activity
- **Relevance to Cognitive Impairment**: Amplitude changes indicate cardiac contractility and electrical conduction affecting brain perfusion

### 2. Time-Domain Heart Rate Variability Features

Time-domain HRV features quantify beat-to-beat variability in RR intervals, providing gold-standard measures of autonomic adaptability and vagal tone linked to cognitive performance.

#### Mean RR / Mean NN Interval (`HRV_MeanNN`)
- **Definition**: Average interval between consecutive normal heartbeats (baseline metric of cardiac chronotropic control)
- **Formula**: $MeanNN = \frac{1}{N}\sum_{i=1}^{N} RR_i$ (in milliseconds)
- **Physiological Significance**: Indicates average heart rate and baseline cardiac function
- **Relevance to Cognitive Impairment**: Baseline cardiac chronotropic control affects cognitive performance and brain perfusion; altered in autonomic dysfunction

#### Standard Deviation of NN Intervals (`HRV_SDNN`)
- **Definition**: Standard deviation of RR intervals—gold standard for overall HRV and autonomic adaptability
- **Formula**: $SDNN = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(RR_i - MeanNN)^2}$
- **Physiological Significance**: Overall heart rate variability reflecting total autonomic activity
- **Relevance to Cognitive Impairment**: Reduced SDNN indicates autonomic dysfunction affecting cognition; strong predictor of cardiovascular and cognitive health

#### Root Mean Square of Successive Differences (`HRV_RMSSD`)
- **Definition**: Root mean square of successive RR differences—primary marker of vagally-mediated (parasympathetic) changes
- **Formula**: $RMSSD = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$
- **Physiological Significance**: Short-term heart rate variability primarily reflecting parasympathetic activity
- **Relevance to Cognitive Impairment**: Important for cognitive flexibility, attention, and stress recovery; key marker of parasympathetic function

#### Standard Deviation of Successive Differences (`HRV_SDSD`)
- **Definition**: Standard deviation of differences between consecutive RR intervals
- **Formula**: $SDSD = \sqrt{\frac{1}{N-2}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i - \overline{\Delta RR})^2}$
- **Physiological Significance**: Another measure of short-term heart rate variability
- **Relevance to Cognitive Impairment**: Complements RMSSD for parasympathetic assessment; provides additional beat-to-beat variability information

#### Coefficient of Variation (`HRV_CVNN`, `HRV_CVSD`)
- **Definition**: Relative variability measures
- **Formula**: $CVNN = \frac{SDNN}{MeanNN}$, $CVSD = \frac{RMSSD}{MeanNN}$
- **Physiological Significance**: Normalized measures of heart rate variability
- **Relevance to Cognitive Impairment**: Relative variability may be more stable across individuals; useful for comparing HRV across different heart rate ranges

#### Median NN Interval (`HRV_MedianNN`)
- **Definition**: Median of RR intervals
- **Formula**: $MedianNN = \text{median}(RR_i)$
- **Physiological Significance**: Robust measure of central tendency, less affected by outliers
- **Relevance to Cognitive Impairment**: May reflect stable cardiac rhythm patterns; MedianNN is less sensitive to ectopic beats than MeanNN

#### Median Absolute Deviation (`HRV_MadNN`)
- **Definition**: Median absolute deviation of NN intervals
- **Formula**: $MadNN = \text{median}(|RR_i - \text{median}(RR_j)|)$
- **Physiological Significance**: Robust measure of variability, resistant to outliers
- **Relevance to Cognitive Impairment**: Indicates consistency in heart rate patterns; MadNN provides robust variability assessment resistant to outliers

#### Interquartile Range (`HRV_IQRNN`)
- **Definition**: Difference between 75th and 25th percentiles of RR intervals
- **Formula**: $IQRNN = Q_3 - Q_1$
- **Physiological Significance**: Reflects the dispersion of heart rate distribution
- **Relevance to Cognitive Impairment**: May indicate range of cardiac responsiveness; IQRNN indicates the spread of heart rate values

#### Percentile Features (`HRV_Prc20NN`, `HRV_Prc80NN`)
- **Definition**: 20th and 80th percentiles of NN intervals
- **Formula**: $Prc20NN = P_{20}(RR_i)$, $Prc80NN = P_{80}(RR_i)$
- **Physiological Significance**: Indicate the range of heart rate values
- **Relevance to Cognitive Impairment**: May reflect cardiac response range to cognitive demands; percentile measures provide distribution information

#### Percentage of NN50 (`HRV_pNN50`, `HRV_pNN20`)
- **Definition**: Percentage of successive RR differences >50 ms (pNN50) or >20 ms (pNN20)
- **Formula**: $pNN50 = \frac{count(|RR_{i+1} - RR_i| > 50\text{ ms})}{N-1} \times 100$
- **Physiological Significance**: Closely correlated with RMSSD; indicates rapid vagal changes
- **Relevance to Cognitive Impairment**: Linked to cognitive performance and stress response; key parasympathetic function marker

#### Min/Max NN Intervals (`HRV_MinNN`, `HRV_MaxNN`)
- **Definition**: Minimum and maximum RR intervals
- **Formula**: $MinNN = \min(RR_i)$, $MaxNN = \max(RR_i)$
- **Physiological Significance**: Extreme heart rate values indicating cardiac range
- **Relevance to Cognitive Impairment**: May reflect cardiac adaptability to cognitive demands; Min/Max NN indicate cardiac reserve and responsiveness

#### Triangular Interpolation of NN Histogram (`HRV_TINN`)
- **Definition**: Width (baseline) of the minimum square difference triangular interpolation of the NN interval histogram
- **Formula**: $TINN = \text{width of best-fit triangle}$ (ms); obtained by minimizing $\sum (y_i - y_{tri})^2$ over triangle parameters
- **Physiological Significance**: Captures geometric distribution of heart rate variability; reflects overall autonomic adaptability
- **Relevance to Cognitive Impairment**: TINN reduction may indicate autonomic dysfunction and cognitive decline

#### Triangular Index (`HRV_Triangular_Index`)
- **Definition**: Total number of NN intervals divided by the height of the RR interval histogram (mode)
- **Formula**: $TriangularIndex = \frac{N}{max(h(k))}$ where $h(k)$ is histogram bin count
- **Physiological Significance**: Robust measure of overall variability; less sensitive to artifacts than SDNN
- **Relevance to Cognitive Impairment**: Provides geometric HRV assessment complementary to time-domain measures; altered in cognitive impairment

#### Heart Rate Triangle Index (`HRV_HTI`)
- **Definition**: Alternative geometric measure: $HTI = N / (MaxNN - MinNN)$
- **Formula**: $HTI = \frac{N}{MaxNN - MinNN}$
- **Physiological Significance**: Overall heart rate variability geometric measure
- **Relevance to Cognitive Impairment**: May reflect overall cardiac health affecting cognition; complements TINN and Triangular Index

#### Ratio Features (`HRV_SDRMSSD`)
- **Definition**: Ratio of SDNN to RMSSD
- **Formula**: $SDRMSSD = \frac{SDNN}{RMSSD}$
- **Physiological Significance**: Balance between overall and short-term variability
- **Relevance to Cognitive Impairment**: May indicate different aspects of autonomic function; SDRMSSD reflects the relationship between overall and short-term variability

### 3. Frequency-Domain Heart Rate Variability Features

Frequency-domain HRV quantifies the distribution of heart rate oscillations across frequency bands, reflecting sympathovagal balance, vagal tone, and overall autonomic capacity—all relevant to cognitive load and stress response.

#### Very Low Frequency Power (`HRV_VLF`)
- **Frequency Range**: 0.0033–0.04 Hz
- **Formula**: $VLF = \sum_{f=0.0033}^{0.04} P(f)$ where $P(f)$ is power spectral density
- **Physiological Significance**: Linked to thermoregulation, hormonal cycles, renin-angiotensin system, and slow autonomic mechanisms
- **Relevance to Cognitive Impairment**: VLF alterations may reflect long-term regulatory dysfunction; associated with cognitive decline in longitudinal studies
- **Note**: Reliable VLF estimation typically requires longer recordings (≥5 min; Task Force recommends 24 h for full VLF interpretation)

#### Low Frequency Power (`HRV_LF`)
- **Frequency Range**: 0.04–0.15 Hz
- **Formula**: $LF = \sum_{f=0.04}^{0.15} P(f)$ where $P(f)$ is power spectral density
- **Physiological Significance**: Reflects a mix of sympathetic and parasympathetic modulation (baroreceptor activity)
- **Relevance to Cognitive Impairment**: Associated with cognitive effort, attention, and mental load; LF changes indicate autonomic nervous system activity during cognitive tasks

#### High Frequency Power (`HRV_HF`)
- **Frequency Range**: 0.15–0.4 Hz
- **Formula**: $HF = \sum_{f=0.15}^{0.4} P(f)$
- **Physiological Significance**: Pure index of vagal (parasympathetic) modulation and respiration
- **Relevance to Cognitive Impairment**: Important for cognitive recovery, relaxation, and attention; key parasympathetic function indicator; reduced in cognitive impairment

#### Very High Frequency Power (`HRV_VHF`)
- **Frequency Range**: 0.4–1.0 Hz (optional; not in standard Task Force bands)
- **Formula**: $VHF = \sum_{f=0.4}^{1.0} P(f)$
- **Physiological Significance**: Physiological significance not fully established; may reflect noise or high-frequency autonomic components
- **Relevance to Cognitive Impairment**: Often excluded from standard analysis; may provide supplementary information in research settings

#### Total Power (`HRV_TP`)
- **Definition**: Sum of energy in all frequency bands (typically VLF + LF + HF for 0.0033–0.4 Hz)
- **Formula**: $TP = VLF + LF + HF$ or $TP = \sum_{f} P(f)$
- **Physiological Significance**: Indicates total variance and overall autonomic capacity
- **Relevance to Cognitive Impairment**: Reflects overall cardiovascular health and autonomic reserve; reduced TP associated with cognitive decline

#### LF/HF Ratio (`HRV_LFHF`)
- **Definition**: Ratio of LF to HF power—represents sympathovagal balance
- **Formula**: $LF/HF = \frac{LF}{HF}$
- **Physiological Significance**: Sympathovagal balance indicator
- **Relevance to Cognitive Impairment**: Indicator of stress, cognitive load, and autonomic balance; elevated LF/HF linked to cognitive impairment and dementia

#### Normalized LF Power (`HRV_LFn`)
- **Definition**: LF power normalized by total power minus VHF
- **Formula**: $LF_n = \frac{LF}{TP - VHF}$
- **Physiological Significance**: Relative sympathetic activity measure
- **Relevance to Cognitive Impairment**: May indicate cognitive stress levels and effort; normalized measures reduce individual differences

#### Normalized HF Power (`HRV_HFn`)
- **Definition**: HF power normalized by total power minus VHF
- **Formula**: $HF_n = \frac{HF}{TP - VHF}$
- **Physiological Significance**: Relative parasympathetic activity measure
- **Relevance to Cognitive Impairment**: Associated with cognitive recovery and attention; normalized HF provides relative parasympathetic assessment

#### Natural Logarithm of HF (`HRV_LnHF`)
- **Definition**: Natural logarithm of high-frequency power
- **Formula**: $LnHF = \ln(HF)$
- **Physiological Significance**: Used to reduce data skewness and improve normality
- **Relevance to Cognitive Impairment**: Provides normalized measure for statistical analysis; log transformation improves statistical properties for modeling

### 4. Non-Linear Heart Rate Variability Features

Nonlinear HRV features capture complex dynamics, fractal properties, and irregularity of heart rate that linear measures miss—relevant to physiological adaptability and cognitive health.

#### Poincaré Plot Features

##### SD1 (`HRV_SD1`)
- **Definition**: Width of the Poincaré plot ellipse (perpendicular to line of identity); correlates with short-term HRV
- **Formula**: $SD1 = \frac{RMSSD}{\sqrt{2}} = \frac{SDSD}{\sqrt{2}}$ (equivalent formulations)
- **Physiological Significance**: Short-term heart rate variability; primary marker of parasympathetic activity
- **Relevance to Cognitive Impairment**: Parasympathetic activity affecting cognitive performance; reduced SD1 in cognitive impairment

##### SD2 (`HRV_SD2`)
- **Definition**: Length of the Poincaré plot ellipse (along line of identity); correlates with long-term HRV
- **Formula**: $SD2 = \sqrt{2 \cdot SDNN^2 - SD1^2}$
- **Physiological Significance**: Long-term heart rate variability; correlates with sympathetic/global activity
- **Relevance to Cognitive Impairment**: Overall autonomic function affecting cognitive processes; reflects long-term autonomic regulation

##### SD1/SD2 Ratio (`HRV_SD1SD2`)
- **Definition**: Ratio of SD1 to SD2
- **Formula**: $SD1/SD2 = \frac{SD1}{SD2}$
- **Physiological Significance**: Balance between short and long-term variability
- **Relevance to Cognitive Impairment**: May indicate autonomic balance affecting cognition; SD1/SD2 ratio indicates short-term vs long-term variability balance

##### Poincaré Plot Area (`HRV_S`)
- **Definition**: Area of the Poincaré plot
- **Formula**: $S = \pi \cdot SD1 \cdot SD2$
- **Physiological Significance**: Overall dynamics of heart rate variability
- **Relevance to Cognitive Impairment**: May reflect overall cardiac health affecting brain function; Poincaré area indicates overall HRV dynamics

#### Complexity Indices

##### Complexity Index (`HRV_CSI`)
- **Definition**: SD2/SD1 ratio, complexity measure
- **Formula**: $CSI = \frac{SD2}{SD1}$
- **Physiological Significance**: Complexity of heart rate variability patterns
- **Relevance to Cognitive Impairment**: May indicate cognitive complexity and adaptability; CSI reflects heart rate pattern complexity

##### Complexity Variability Index (`HRV_CVI`)
- **Definition**: Logarithmic product of SD1 and SD2
- **Formula**: $CVI = \log(SD1 \cdot SD2)$
- **Physiological Significance**: Complexity and irregularity of heart rate sequence
- **Relevance to Cognitive Impairment**: May indicate cognitive flexibility and adaptability; CVI provides logarithmic complexity measure

##### Modified Complexity Index (`HRV_CSI_Modified`)
- **Definition**: Modified complexity index with different calculation approach
- **Physiological Significance**: Alternative complexity assessment method
- **Relevance to Cognitive Impairment**: May provide additional complexity information; alternative complexity measure

#### Fractal Analysis Features

##### Detrended Fluctuation Analysis Alpha 1 (`HRV_DFA_alpha1`)
- **Definition**: Short-term scaling exponent in DFA; describes fractal correlation at small time scales (typically 4–16 beats)
- **Formula**: $F(n) \propto n^{\alpha_1}$; $\alpha_1$ from linear fit of $\log F(n)$ vs $\log n$ for $n \in [4, 16]$
- **Physiological Significance**: Short-range fractal correlations; reflects beat-to-beat and short-term regulatory dynamics
- **Relevance to Cognitive Impairment**: Altered α1 indicates impaired short-term autonomic regulation; may precede cognitive decline

##### Detrended Fluctuation Analysis Alpha 2 (`HRV_DFA_alpha2`)
- **Definition**: Long-term scaling exponent in DFA; describes fractal correlation at large time scales (typically 16–64 beats)
- **Formula**: $F(n) \propto n^{\alpha_2}$; $\alpha_2$ from linear fit of $\log F(n)$ vs $\log n$ for $n \in [16, 64]$
- **Physiological Significance**: Long-range fractal correlations; describes fractal correlation properties of heart rate dynamics
- **Relevance to Cognitive Impairment**: α2 reflects long-term regulatory complexity; reduced in dementia and MCI; useful for cognitive assessment

##### Multifractal Detrended Fluctuation Analysis Features

###### MFDFA Alpha 2 Width (`HRV_MFDFA_alpha2_Width`)
- **Definition**: Width of the multifractal spectrum
- **Formula**: $Width = \alpha_{max} - \alpha_{min}$
- **Physiological Significance**: Multifractal characteristics of heart rate signals
- **Relevance to Cognitive Impairment**: Indicates complexity of autonomic regulation; MFDFA width reflects multifractal diversity

###### MFDFA Alpha 2 Peak (`HRV_MFDFA_alpha2_Peak`)
- **Definition**: Peak value in the multifractal spectrum
- **Formula**: $Peak = f(\alpha_{peak})$
- **Physiological Significance**: Strongest multifractal characteristic of heart rate signal
- **Relevance to Cognitive Impairment**: May reflect dominant cognitive processing patterns; MFDFA peak indicates primary multifractal behavior

###### MFDFA Alpha 2 Mean (`HRV_MFDFA_alpha2_Mean`)
- **Definition**: Mean value in the multifractal spectrum
- **Formula**: $Mean = \frac{1}{n}\sum_{i=1}^{n} f(\alpha_i)$
- **Physiological Significance**: Overall multifractal characteristics
- **Relevance to Cognitive Impairment**: May indicate average cognitive complexity; MFDFA mean reflects overall multifractal properties

###### MFDFA Alpha 2 Max (`HRV_MFDFA_alpha2_Max`)
- **Definition**: Maximum value in the multifractal spectrum
- **Formula**: $Max = \max(f(\alpha_i))$
- **Physiological Significance**: Maximum multifractal characteristic
- **Relevance to Cognitive Impairment**: May indicate peak cognitive complexity; MFDFA max indicates maximum multifractal strength

###### MFDFA Alpha 2 Delta (`HRV_MFDFA_alpha2_Delta`)
- **Definition**: Delta value in the multifractal spectrum
- **Formula**: $Delta = \alpha_{max} - \alpha_{min}$
- **Physiological Significance**: Diversity of heart rate signals
- **Relevance to Cognitive Impairment**: May indicate cognitive diversity and flexibility; MFDFA delta reflects multifractal range

###### MFDFA Alpha 2 Asymmetry (`HRV_MFDFA_alpha2_Asymmetry`)
- **Definition**: Asymmetry in the multifractal spectrum
- **Formula**: $Asymmetry = \frac{|\alpha_{left} - \alpha_{right}|}{|\alpha_{left} + \alpha_{right}|}$
- **Physiological Significance**: Directional behavior of heart rate variability
- **Relevance to Cognitive Impairment**: May indicate pathological autonomic patterns; MFDFA asymmetry reflects directional multifractal behavior

###### MFDFA Alpha 2 Fluctuation (`HRV_MFDFA_alpha2_Fluctuation`)
- **Definition**: Fluctuation in the multifractal spectrum
- **Formula**: $Fluctuation = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(f(\alpha_i) - \overline{f(\alpha)})^2}$
- **Physiological Significance**: Stability of heart rate signals
- **Relevance to Cognitive Impairment**: May indicate cognitive stability; MFDFA fluctuation reflects multifractal stability

###### MFDFA Alpha 2 Increment (`HRV_MFDFA_alpha2_Increment`)
- **Definition**: Increment in the multifractal spectrum
- **Formula**: $Increment = \sum_{i=1}^{n-1}|f(\alpha_{i+1}) - f(\alpha_i)|$
- **Physiological Significance**: Trend of heart rate variability
- **Relevance to Cognitive Impairment**: May indicate cognitive trend patterns; MFDFA increment reflects multifractal trends

### 5. Cardiorespiratory Coupling

Cardiorespiratory coupling quantifies the temporal coordination between cardiac and respiratory systems—a direct measure of vagal tone and breathing–heart rate synchronization relevant to cognitive regulation and stress response.

#### RSA Amplitude (`RSA_Amplitude`)
- **Definition**: Respiratory Sinus Arrhythmia magnitude—the variation in heart rate (or RR interval) synchronized with respiration
- **Formula**: $RSA = \max(RR_{insp}) - \min(RR_{exp})$ or spectral peak in HF band; alternatively from EDR-RR coherence
- **Physiological Significance**: Direct measure of vagal tone and breathing–heart rate synchronization
- **Relevance to Cognitive Impairment**: RSA reflects parasympathetic reserve; reduced RSA associated with autonomic dysfunction and cognitive decline; key biomarker when respiration is available

#### ECG-Derived Respiration (`EDR`)
- **Definition**: Surrogate respiration signal extracted from ECG (e.g., R-peak amplitude modulation, QRS area, or thoracic impedance proxy)
- **Formula**: EDR derived from amplitude/area changes of R-wave or QRS complex across respiratory cycle
- **Physiological Significance**: Surrogate for respiratory effort when direct breath signal is absent
- **Relevance to Cognitive Impairment**: Enables RSA and phase synchronization analysis from ECG alone; useful for cognitive assessment when respiratory recording is unavailable

#### Phase Synchronization (`Phase_Sync_ECG_Resp`)
- **Definition**: Phase locking value (PLV) or coherence between ECG R-peaks and respiration phase
- **Formula**: $PLV = |\frac{1}{N}\sum_{k=1}^{N} e^{j(\phi_{RR}(k) - \phi_{resp}(k))}|$ where $\phi$ are instantaneous phases
- **Physiological Significance**: Quantifies the temporal coordination between cardiac and respiratory systems
- **Relevance to Cognitive Impairment**: Reduced phase synchronization may indicate impaired brainstem-mediated cardiorespiratory coupling; altered in stress and cognitive load

### 6. Entropy-Based Features

Entropy measures quantify regularity, irregularity, and information content of RR interval sequences—reduced entropy indicates decreased physiological adaptability in cognitive impairment.

#### Approximate Entropy (`HRV_ApEn`)
- **Definition**: Measure of regularity in RR interval sequence; indicates predictability of heart rate fluctuations
- **Formula**: $ApEn(m,r,N) = \phi^m(r) - \phi^{m+1}(r)$ where $\phi^m(r) = \frac{1}{N-m+1}\sum_{i=1}^{N-m+1}\ln C_i^m(r)$
- **Physiological Significance**: Quantifies complexity of heart rate patterns
- **Relevance to Cognitive Impairment**: Reduced ApEn indicates decreased autonomic complexity; decreased in various pathological conditions including cognitive impairment

#### Sample Entropy (`HRV_sampEn`)
- **Definition**: Improved measure of signal complexity; reflects heartbeat irregularity and physiological adaptability
- **Formula**: $SampEn(m,r,N) = -\ln\frac{A}{B}$ where $A$ and $B$ are template match counts
- **Physiological Significance**: Better measure of heart rate regularity; more robust than ApEn
- **Relevance to Cognitive Impairment**: More reliable for cognitive assessment; reduced in dementia and MCI

#### Shannon Entropy (`HRV_shanEn`)
- **Definition**: Information content of RR interval distribution
- **Formula**: $H = -\sum_{i} p_i \log p_i$ where $p_i$ is probability of interval $i$
- **Physiological Significance**: Measures uncertainty in heart rate patterns
- **Relevance to Cognitive Impairment**: May indicate cognitive load and stress; Shannon entropy reflects information content of RR distribution

#### Fuzzy Entropy (`HRV_FuzzyEn`)
- **Definition**: Entropy measure using fuzzy logic
- **Formula**: $FuzzyEn(m,r,N) = -\ln\frac{\Phi^m(r)}{\Phi^{m+1}(r)}$
- **Physiological Significance**: Improved entropy calculation with fuzzy logic
- **Relevance to Cognitive Impairment**: Better complexity assessment for cognitive applications; fuzzy entropy provides robust complexity measures

#### Multi-Scale Entropy (`HRV_MSEn`)
- **Definition**: Entropy analysis across different time scales
- **Formula**: $MSEn(\tau) = SampEn(\tau)$ for scale factor $\tau$
- **Physiological Significance**: Analyzes complexity across different time scales
- **Relevance to Cognitive Impairment**: May indicate multi-scale cognitive complexity; MSEn provides scale-dependent complexity assessment

#### Composite Multi-Scale Entropy (`HRV_CMSEn`)
- **Definition**: Improvement to multi-scale entropy
- **Formula**: $CMSEn(\tau) = -\ln\frac{\langle A(\tau) \rangle}{\langle B(\tau) \rangle}$
- **Physiological Significance**: Increased sensitivity to heart rate variability analysis
- **Relevance to Cognitive Impairment**: More sensitive cognitive complexity assessment; CMSEn provides enhanced multi-scale analysis

#### Refined Composite Multi-Scale Entropy (`HRV_RCMSEn`)
- **Definition**: Further improvement to multi-scale entropy calculation
- **Formula**: $RCMSEn(\tau) = -\ln\frac{\sum_{j=1}^{\tau}A_j(\tau)}{\sum_{j=1}^{\tau}B_j(\tau)}$
- **Physiological Significance**: Most refined multi-scale entropy measure
- **Relevance to Cognitive Impairment**: Most accurate cognitive complexity assessment; RCMSEn provides the most robust multi-scale analysis

### 7. Heart Rate Asymmetry Features (HRA)

#### Guzik Index (`HRV_GI`)
- **Definition**: Asymmetry of NN intervals in Poincaré plot between acceleration and deceleration
- **Formula**: $GI = \frac{4\sum_{i=1}^{N-1}(RR_i \cdot RR_{i+1})}{(RR_1^2 + RR_N^2 + 2\sum_{i=1}^{N-1}RR_i^2)}$
- **Physiological Significance**: Asymmetry between heart rate acceleration and deceleration
- **Relevance to Cognitive Impairment**: May indicate cognitive processing asymmetry; GI reflects autonomic asymmetry

#### Slope Index (`HRV_SI`)
- **Definition**: Degree of slope in Poincaré plot shape
- **Formula**: $SI = \frac{\sum_{i=1}^{N-1}(RR_i - \overline{RR})(RR_{i+1} - \overline{RR})}{\sum_{i=1}^{N-1}(RR_i - \overline{RR})^2}$
- **Physiological Significance**: Asymmetry in heart rate variability patterns
- **Relevance to Cognitive Impairment**: May indicate cognitive response asymmetry; SI reflects directional heart rate patterns

#### Area Index (`HRV_AI`)
- **Definition**: Asymmetry in heart rate variability by calculating area of specific regions
- **Formula**: $AI = \frac{Area_{above} - Area_{below}}{Area_{above} + Area_{below}}$
- **Physiological Significance**: Asymmetry evaluation in Poincaré plot regions
- **Relevance to Cognitive Impairment**: May indicate cognitive processing asymmetry; AI provides geometric asymmetry assessment

#### Porta Index (`HRV_PI`)
- **Definition**: Ratio of increasing to decreasing RR intervals
- **Formula**: $PI = \frac{count(RR_{i+1} > RR_i)}{count(RR_{i+1} < RR_i)}$
- **Physiological Significance**: Asymmetry in heart rate variability direction
- **Relevance to Cognitive Impairment**: May indicate cognitive response directionality; PI reflects directional heart rate asymmetry

#### Acceleration/Deceleration Contributions

##### Short-term Contributions (`HRV_C1d`, `HRV_C1a`)
- **Definition**: Contribution of deceleration/acceleration to short-term variability
- **Formula**: $C1d = \frac{SD1_d}{SD1}$, $C1a = \frac{SD1_a}{SD1}$
- **Physiological Significance**: Sympathetic vs parasympathetic contributions
- **Relevance to Cognitive Impairment**: May indicate cognitive stress vs recovery patterns; C1d/C1a reflect autonomic balance components

##### Short-term Standard Deviations (`HRV_SD1d`, `HRV_SD1a`)
- **Definition**: Short-term standard deviation of deceleration/acceleration
- **Formula**: $SD1d = SD1 \cdot C1d$, $SD1a = SD1 \cdot C1a$
- **Physiological Significance**: Separate short-term variability components
- **Relevance to Cognitive Impairment**: May indicate separate cognitive processing components; SD1d/SD1a provide component-specific variability

##### Long-term Contributions (`HRV_C2d`, `HRV_C2a`)
- **Definition**: Contribution of deceleration/acceleration to long-term variability
- **Formula**: $C2d = \frac{SD2_d}{SD2}$, $C2a = \frac{SD2_a}{SD2}$
- **Physiological Significance**: Long-term autonomic contributions
- **Relevance to Cognitive Impairment**: May indicate long-term cognitive patterns; C2d/C2a reflect long-term autonomic components

##### Long-term Standard Deviations (`HRV_SD2d`, `HRV_SD2a`)
- **Definition**: Long-term standard deviation of deceleration/acceleration
- **Formula**: $SD2d = SD2 \cdot C2d$, $SD2a = SD2 \cdot C2a$
- **Physiological Significance**: Separate long-term variability components
- **Relevance to Cognitive Impairment**: May indicate separate long-term cognitive patterns; SD2d/SD2a provide long-term component variability

##### Total Contributions (`HRV_Cd`, `HRV_Ca`)
- **Definition**: Total contribution of deceleration/acceleration to variability
- **Formula**: $Cd = \frac{SDNN_d}{SDNN}$, $Ca = \frac{SDNN_a}{SDNN}$
- **Physiological Significance**: Total autonomic contributions
- **Relevance to Cognitive Impairment**: May indicate total cognitive processing components; Cd/Ca reflect total autonomic balance

##### Total Standard Deviations (`HRV_SDNNd`, `HRV_SDNNa`)
- **Definition**: Total standard deviation of deceleration/acceleration
- **Formula**: $SDNNd = SDNN \cdot Cd$, $SDNNa = SDNN \cdot Ca$
- **Physiological Significance**: Total variability components
- **Relevance to Cognitive Impairment**: May indicate total cognitive variability components; SDNNd/SDNNa provide total component variability

### 8. Heart Rate Fragmentation Features (HRF)

#### Percentage of Inflection Points (`HRV_PIP`)
- **Definition**: Percentage of inflection points in NN interval sequence
- **Formula**: $PIP = \frac{count(\text{inflection points})}{N-2} \times 100$
- **Physiological Significance**: Indicates fragmentation in heart rate patterns
- **Relevance to Cognitive Impairment**: May indicate autonomic dysfunction affecting cognition; PIP reflects heart rate pattern fragmentation; increased in adverse cardiovascular outcomes

#### Inverse of Average Length of Segments (`HRV_IALS`)
- **Definition**: Measure of segment length in NN interval sequence
- **Formula**: $IALS = \frac{1}{\overline{L}}$ where $\overline{L}$ is average segment length
- **Physiological Significance**: Indicates continuity of heart rate patterns
- **Relevance to Cognitive Impairment**: May reflect cognitive stability and continuity; IALS reflects heart rate pattern continuity

#### Percentage of Short Segments (`HRV_PSS`)
- **Definition**: Percentage of short segments in NN interval sequence
- **Formula**: $PSS = \frac{count(\text{short segments})}{total \text{ segments}} \times 100$
- **Physiological Significance**: Proportion of transient patterns
- **Relevance to Cognitive Impairment**: May indicate transient cognitive processing; PSS reflects transient heart rate patterns

#### Percentage of Alternating Segments (`HRV_PAS`)
- **Definition**: Percentage of alternating acceleration/deceleration segments
- **Formula**: $PAS = \frac{count(\text{alternating segments})}{total \text{ segments}} \times 100$
- **Physiological Significance**: Proportion of alternating patterns
- **Relevance to Cognitive Impairment**: May indicate autonomic rhythm affecting cognition; PAS reflects alternating heart rate patterns

### 9. Fractal and Complexity Features

#### Correlation Dimension (`HRV_CD`)
- **Definition**: Fractal dimension reflecting correlation structure
- **Formula**: $CD = \lim_{r \to 0} \frac{\log C(r)}{\log r}$ where $C(r)$ is correlation integral
- **Physiological Significance**: Fractal characteristics of heart rate time series
- **Relevance to Cognitive Impairment**: May indicate cognitive complexity and structure; CD reflects heart rate correlation structure

#### Higuchi Fractal Dimension (`HRV_HFD`)
- **Definition**: Fractal dimension using Higuchi method
- **Formula**: $HFD = \frac{\log(L(k))}{\log(k)}$ where $L(k)$ is average length
- **Physiological Significance**: Complexity and self-similarity of heart rate time series
- **Relevance to Cognitive Impairment**: May indicate cognitive complexity and self-similarity; HFD provides robust fractal dimension measure; reduced in dementia

#### Katz Fractal Dimension (`HRV_KFD`)
- **Definition**: Fractal dimension using Katz method
- **Formula**: $KFD = \frac{\log(N)}{\log(\frac{d}{L})}$ where $d$ is distance, $L$ is total length
- **Physiological Significance**: Complexity of heart rate time series
- **Relevance to Cognitive Impairment**: May indicate overall cognitive complexity; KFD provides alternative fractal dimension measure

#### Lempel-Ziv Complexity (`HRV_LZC`)
- **Definition**: Complexity based on pattern repetition
- **Formula**: $LZC = \frac{c(n)}{n/\log_2(n)}$ where $c(n)$ is number of distinct patterns
- **Physiological Significance**: Complexity and pattern diversity in heart rate time series
- **Relevance to Cognitive Impairment**: May indicate cognitive pattern complexity; LZC reflects heart rate pattern complexity; decreased in MCI

## Implementation Pipeline

### Data Preprocessing
1. **R-Peak Detection**: 
   - Pan-Tompkins algorithm
   - Hamilton algorithm
   - Wavelet-based detection
2. **P-QRS-T Delineation** (for morphological features): 
   - Wavelet-based or template-matching approaches for P, QRS, T onset/offset
   - Required for PR interval, QRS duration, QT/JT, P/T amplitude, ST segment
3. **Artifact Removal**: 
   - Ectopic beat detection and correction
   - Outlier removal (>3 standard deviations)
   - Interpolation of missing beats
4. **Quality Control**: 
   - Minimum data length (>5 minutes for reliable HRV; shorter for morphology)
   - Artifact rate (<5% of beats)
   - Stationarity checks

### Feature Extraction Pipeline
1. **ECG Preprocessing**: Filtering, baseline correction, R-peak detection (e.g., Pan-Tompkins, Hamilton)
2. **Morphological Extraction**: P-QRS-T delineation for PR, QRS, QT/JT, P/T amplitude, ST segment
3. **RR Interval Extraction**: Detect R-peaks and compute NN intervals
4. **Artifact Correction**: Ectopic beat detection/correction, outlier removal, interpolation
5. **Feature Computation**: Time-domain, frequency-domain, nonlinear, cardiorespiratory coupling, entropy, HRA, HRF, fractal features
6. **Validation**: Check feature ranges and consistency
7. **Normalization**: Apply appropriate scaling for each feature type

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

The comprehensive ECG feature set described in this document provides a robust foundation for cognitive impairment assessment through autonomic nervous system and neurocardiac regulation evaluation. The features span six major categories aligned with the expanded ECG/HRV taxonomy:

1. **Morphological features**: PR interval, QRS duration, QT/JT interval, P/T wave amplitude, ST segment, skewness, kurtosis—reflecting atrioventricular conduction and cardiac repolarization linked to neurocardiac regulation.
2. **Time-domain HRV**: Mean RR, SDNN, RMSSD, pNN50, TINN, Triangular Index—gold-standard measures of autonomic adaptability and vagal tone.
3. **Frequency-domain HRV**: VLF, LF, HF, LF/HF, Total Power—sympathovagal balance and parasympathetic modulation.
4. **Nonlinear HRV**: Poincaré SD1/SD2, DFA α1/α2, SampEn, ApEn—fractal and complexity properties of heart rate dynamics.
5. **Cardiorespiratory coupling**: RSA amplitude, EDR, Phase synchronization—direct measures of vagal tone and breathing–heart rate coordination.
6. **Entropy, HRA, HRF, and fractal features**: Multi-scale entropy, heart rate asymmetry, fragmentation, and fractal dimension—complementary complexity and pattern measures.

When combined with appropriate signal processing techniques and clinical validation, these features can serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders. The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of autonomic nervous system activity.

The extensive literature support and open-source code libraries provide researchers and clinicians with the tools necessary to implement and validate these features in their own studies, contributing to the advancement of cognitive health assessment through cardiovascular monitoring.
