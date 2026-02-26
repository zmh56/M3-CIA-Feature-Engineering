# Speech Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of speech features extracted for cognitive impairment assessment in the M3-CIA framework. Speech features are extracted from picture description tasks (specifically the Cookie Theft picture from the Boston Diagnostic Aphasia Examination) and categorized into five main domains aligned with the expanded speech taxonomy: (1) spectral features, (2) prosodic features, (3) fluency & temporal features, (4) voice quality features, and (5) semantic features. These features capture vocal tract characteristics, prosody, speech motor planning, lexical/syntactic complexity, and semantic content—all closely linked to cognitive performance and brain health.

## Overview

Speech features are extracted from the Cookie Theft picture description task, a standardized cognitive assessment tool that requires participants to describe a complex scene. This task captures the acoustic properties, paralinguistic characteristics, and semantic content of speech production under cognitive load. These features serve as objective biomarkers for detecting and monitoring cognitive impairment through speech analysis, providing non-invasive assessment of cognitive function.

## Feature Categories and Descriptions

### 1. Spectral Features

Spectral features capture vocal tract spectral characteristics and resonance patterns, providing markers of altered phonation and articulatory precision associated with cognitive impairment.

#### MFCC (1–13) (`mfcc1-13_sma3_*`)
- **Definition**: Mel-frequency cepstral coefficients computed via Fourier transform and mel-scale filtering
- **Formula**: $MFCC_i = \sum_{k=1}^{K} \log(|X_{mel}(k)|) \cos\left(\frac{i(k-0.5)\pi}{K}\right)$ where $X_{mel}$ is mel-filterbank output
- **Physiological Significance**: Captures vocal tract spectral envelope; provides markers of altered phonation patterns
- **Relevance to Cognitive Impairment**: MFCC changes indicate articulation and phonation deficits in dementia and MCI; first 4–13 coefficients commonly used in eGeMAPS

#### Formants (F1–F3) (`F1`, `F2`, `F3`)
- **Definition**: Spectral peaks of vocal tract resonance calculated through LPC (Linear Predictive Coding) analysis
- **Formula**: $F_i = \text{ith pole of LPC spectrum}$; typically extracted via autocorrelation or covariance method
- **Physiological Significance**: Indicate articulatory precision and resonance patterns associated with language production
- **Relevance to Cognitive Impairment**: Formant shifts and reduced precision indicate articulation deficits in dementia and MCI; F1 (tongue height), F2 (front-back), F3 (lip rounding) reflect speech motor control

#### Spectral Centroid (`spectral_centroid`, `spectralCentroid_sma3_*`)
- **Definition**: The "center of mass" of the spectrum (spectral brightness)
- **Formula**: $SC = \frac{\sum_{k=1}^{K} f_k \cdot |X(k)|}{\sum_{k=1}^{K} |X(k)|}$ where $f_k$ is frequency bin
- **Physiological Significance**: Reflects high-frequency energy distribution; lower centroid indicates duller voice
- **Relevance to Cognitive Impairment**: Reflects high-frequency energy loss often linked to articulatory slurring; reduced in cognitive impairment

#### Spectral Flux (`spectralFlux_sma3_*`, `spectralFluxV_sma3nz_*`)
- **Definition**: Rate of change of the power spectrum between consecutive frames
- **Formula**: $SF_t = \sum_{k=1}^{K} |X_{t+1}(k) - X_t(k)|$ or normalized variant
- **Physiological Significance**: Indicates dynamic changes in spectral shape; related to speech motor planning
- **Relevance to Cognitive Impairment**: Altered spectral flux may indicate impaired articulatory transitions and motor planning in cognitive disorders

### 2. Prosodic Features

Prosodic features capture fundamental frequency, intensity, and intonation patterns—reflecting vocal control, affective modulation, and expressive prosody that diminish in cognitively impaired speech.

#### F0 (Mean/Std) (`F0semitoneFrom27.5Hz_sma3nz_amean`, `F0semitoneFrom27.5Hz_sma3nz_stddevNorm`)
- **Definition**: Fundamental frequency statistics derived from vocal fold vibration
- **Formula**: $F_0^{semitone} = 12 \times \log_2\left(\frac{F_0}{27.5}\right)$; Mean and Std over voiced frames
- **Physiological Significance**: Reflects vocal control, intonation flatness, and affective modulation
- **Relevance to Cognitive Impairment**: F0 changes associated with Parkinson's disease, depression, and cognitive decline; reduced variability in Alzheimer's disease
- **Specific Features**: `amean`, `stddevNorm`, `percentile20.0/50.0/80.0`, `meanRisingSlope`, `meanFallingSlope`

#### Intensity (`loudness_sma3_*`, RMS)
- **Definition**: Root Mean Square (RMS) amplitude and its dynamic range; perceptual loudness
- **Formula**: $RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$; $Loudness = \sqrt[0.67]{\sum_{k} |X(k)|^{0.67}}$
- **Physiological Significance**: Indicates ability to modulate loudness
- **Relevance to Cognitive Impairment**: Often reduced in apathy or depression; reflects vocal effort and sustained attention capacity

#### Pitch Contour (`F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope`, `meanFallingSlope`)
- **Definition**: Trajectory of F0 over time (e.g., rising/falling slopes)
- **Formula**: $Slope = \frac{\Delta F_0}{\Delta t}$ over voiced segments; contour statistics
- **Physiological Significance**: Captures expressive prosody and intonation patterns
- **Relevance to Cognitive Impairment**: Expressive prosody diminishes in cognitively impaired speech; flat pitch contour in dementia

##### Fundamental Frequency Range (`F0_range`)
- **Definition**: Difference between maximum and minimum F0 values
- **Formula**: $F_0^{range} = F_0^{max} - F_0^{min}$
- **Physiological Significance**: Reflects vocal range and flexibility
- **Relevance to Cognitive Impairment**: F0 range reduced in cognitive disorders; indicates cognitive flexibility and expression

##### Fundamental Frequency Slope (`F0_slope`)
- **Definition**: Rate of change of F0 over time
- **Formula**: $F_0^{slope} = \frac{dF_0}{dt}$
- **Physiological Significance**: Indicates intonation patterns and prosody
- **Relevance to Cognitive Impairment**: F0 slope changes indicate prosodic deficits; flat slopes in dementia

##### First Formant (`F1`)
- **Definition**: First formant frequency (lowest resonance)
- **Formula**: $F1 = \text{first resonance frequency of vocal tract}$ (LPC)
- **Physiological Significance**: Reflects tongue height and jaw position
- **Relevance to Cognitive Impairment**: F1 changes indicate articulation disorders; altered in dementia

##### Second Formant (`F2`)
- **Definition**: Second formant frequency
- **Formula**: $F2 = \text{second resonance frequency of vocal tract}$ (LPC)
- **Physiological Significance**: Reflects tongue front-back position
- **Relevance to Cognitive Impairment**: F2 changes indicate articulation and vowel production; reduced precision in MCI

##### Third Formant (`F3`)
- **Definition**: Third formant frequency
- **Formula**: $F3 = \text{third resonance frequency of vocal tract}$ (LPC)
- **Physiological Significance**: Reflects tongue tip position and lip rounding
- **Relevance to Cognitive Impairment**: F3 changes indicate articulation disorders

##### Formant Bandwidth (`F1_BW`, `F2_BW`, `F3_BW`)
- **Definition**: Bandwidth of formant frequencies
- **Formula**: $F_i^{BW} = \text{bandwidth of } F_i$
- **Physiological Significance**: Reflects vocal tract damping
- **Relevance to Cognitive Impairment**: Formant bandwidth changes indicate speech quality degradation

#### Loudness / Intensity Features (`loudness_sma3_*`)
- **Definition**: Perceptual loudness measures using spectral analysis; RMS amplitude
- **Formula**: $Loudness = \sqrt[0.67]{\sum_{k=1}^{K} |X(k)|^{0.67}}$; $RMS = \sqrt{\frac{1}{N}\sum x_i^2}$
- **Physiological Significance**: Reflects perceived vocal effort and loudness; ability to modulate loudness
- **Relevance to Cognitive Impairment**: Often reduced in apathy or depression; indicates vocal effort and sustained attention
- **Specific Features**:
  - `loudness_sma3_amean`: Mean loudness
  - `loudness_sma3_stddevNorm`: Normalized standard deviation
  - `loudness_sma3_percentile20.0/50.0/80.0`: Percentile values
  - `loudness_sma3_meanRisingSlope/FallingSlope`: Slope features

##### Zero Crossing Rate (`ZCR`)
- **Definition**: Rate of sign changes in the signal
- **Formula**: $ZCR = \frac{1}{N-1}\sum_{i=1}^{N-1} |sgn(x(i)) - sgn(x(i+1))|$
- **Physiological Significance**: Distinguishes between voiced and unvoiced segments
- **Relevance to Cognitive Impairment**: May indicate speech rhythm and timing; ZCR changes indicate voice quality variations

##### Spectral Rolloff (`spectral_rolloff`)
- **Definition**: Frequency below which 85% of spectral energy is contained
- **Formula**: $SR = \min(f) \text{ such that } \sum_{k=1}^{f} |X(k)| = 0.85 \sum_{k=1}^{K} |X(k)|$
- **Physiological Significance**: Reflects spectral shape and voice quality
- **Relevance to Cognitive Impairment**: Spectral rolloff changes indicate voice quality degradation; high-frequency loss in cognitive impairment

#### Additional Spectral Features (eGeMAPS)

##### MFCC Features (`mfcc1-4_sma3_*`, `mfcc1-13`)
- **Definition**: Mel-frequency cepstral coefficients capturing spectral envelope (via Fourier transform and mel-scale filtering)
- **Formula**: $MFCC_i = \sum_{k=1}^{K} \log(|X_{mel}(k)|) \cos\left(\frac{i(k-0.5)\pi}{K}\right)$
- **Physiological Significance**: Captures vocal tract spectral characteristics; markers of altered phonation patterns
- **Relevance to Cognitive Impairment**: MFCC changes indicate articulation disorders; effective for dementia and MCI detection
- **Specific Features**:
  - `mfcc1-4_sma3_amean`: Mean values of first 4 MFCC coefficients
  - `mfcc1-4_sma3_stddevNorm`: Normalized standard deviations
  - `mfcc1-4V_sma3nz_amean/stddevNorm`: MFCC for voiced segments

##### Spectral Bandwidth (`spectral_bandwidth`)
- **Definition**: Width of the spectrum around the centroid
- **Formula**: $SB = \sqrt{\frac{\sum_{k=1}^{K} (k - SC)^2 \cdot |X(k)|}{\sum_{k=1}^{K} |X(k)|}}$
- **Physiological Significance**: Reflects spectral spread and voice quality
- **Relevance to Cognitive Impairment**: May indicate speech clarity and quality; spectral bandwidth changes indicate voice quality variations

##### Spectral Contrast (`spectral_contrast`)
- **Definition**: Difference between peak and valley in spectral sub-bands
- **Formula**: $SC = \frac{1}{M}\sum_{m=1}^{M} \log\left(\frac{P_m}{V_m}\right)$
- **Physiological Significance**: Reflects spectral dynamics and timbre
- **Relevance to Cognitive Impairment**: May indicate speech expressiveness; spectral contrast changes indicate voice quality

#### 1.5 Voice Quality Features

##### Jitter Features (`jitterLocal_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in fundamental frequency
- **Formula**: $Jitter = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|F_0(i+1) - F_0(i)|}{F_0(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Relevance to Cognitive Impairment**: May indicate neurological control; jitter increased in neurological disorders
- **Specific Features**: `jitterLocal_sma3nz_amean`, `jitterLocal_sma3nz_stddevNorm`

##### Shimmer Features (`shimmerLocaldB_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in amplitude (in dB)
- **Formula**: $Shimmer = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|A(i+1) - A(i)|}{A(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Relevance to Cognitive Impairment**: May indicate neurological control; shimmer increased in neurological disorders
- **Specific Features**: `shimmerLocaldB_sma3nz_amean`, `shimmerLocaldB_sma3nz_stddevNorm`

##### Harmonics-to-Noise Ratio (`HNRdBACF_sma3nz_*`)
- **Definition**: Ratio of harmonic to noise energy using autocorrelation function
- **Formula**: $HNR = 10\log_{10}\left(\frac{P_{harmonic}}{P_{noise}}\right)$
- **Physiological Significance**: Reflects voice quality and clarity
- **Relevance to Cognitive Impairment**: May indicate vocal control and health; HNR decreased in voice disorders
- **Specific Features**: `HNRdBACF_sma3nz_amean`, `HNRdBACF_sma3nz_stddevNorm`

#### 1.6 Spectral Shape Features

##### Spectral Slope Features (`slopeV0-500_sma3nz_*`, `slopeV500-1500_sma3nz_*`)
- **Definition**: Spectral slope in different frequency ranges for voiced segments
- **Formula**: $Slope = \frac{\log(P_{f2}) - \log(P_{f1})}{\log(f2) - \log(f1)}$
- **Physiological Significance**: Reflects spectral shape and voice quality
- **Relevance to Cognitive Impairment**: May indicate speech clarity and quality; spectral slope changes indicate voice quality variations
- **Specific Features**: 
  - `slopeV0-500_sma3nz_amean/stddevNorm`: Slope in 0-500Hz range
  - `slopeV500-1500_sma3nz_amean/stddevNorm`: Slope in 500-1500Hz range

##### Alpha Ratio (`alphaRatioV_sma3nz_*`)
- **Definition**: Ratio of energy in 50-1000Hz to 1-5kHz for voiced segments
- **Formula**: $AlphaRatio = \frac{E_{50-1000Hz}}{E_{1-5kHz}}$
- **Physiological Significance**: Reflects spectral balance and voice quality
- **Relevance to Cognitive Impairment**: May indicate speech clarity and quality; alpha ratio changes indicate voice quality variations

##### Hammarberg Index (`hammarbergIndexV_sma3nz_*`, `hammarbergIndexUV_sma3nz_*`)
- **Definition**: Ratio of maximum energy in 0-2kHz to maximum energy in 2-5kHz
- **Formula**: $HI = \frac{\max(E_{0-2kHz})}{\max(E_{2-5kHz})}$
- **Physiological Significance**: Reflects spectral balance and voice quality
- **Relevance to Cognitive Impairment**: May indicate speech clarity and quality; Hammarberg index changes indicate voice quality variations
- **Specific Features**: 
  - `hammarbergIndexV_sma3nz_amean/stddevNorm`: For voiced segments
  - `hammarbergIndexUV_sma3nz_amean`: For unvoiced segments

#### 1.7 Spectral Flux Features

##### Spectral Flux (`spectralFlux_sma3_*`, `spectralFluxV_sma3nz_*`, `spectralFluxUV_sma3nz_*`)
- **Definition**: Rate of change of spectral energy over time
- **Formula**: $SF = \sum_{k=1}^{K} |X_{t+1}(k) - X_t(k)|$
- **Physiological Significance**: Reflects spectral dynamics and timbre changes
- **Relevance to Cognitive Impairment**: May indicate speech expressiveness and clarity; spectral flux changes indicate voice quality variations
- **Specific Features**: 
  - `spectralFlux_sma3_amean/stddevNorm`: Overall spectral flux
  - `spectralFluxV_sma3nz_amean/stddevNorm`: For voiced segments
  - `spectralFluxUV_sma3nz_amean`: For unvoiced segments

#### 1.8 Segment Statistics Features

##### Voiced/Unvoiced Segment Features
- **Definition**: Statistics of voiced and unvoiced speech segments
- **Physiological Significance**: Reflects speech rhythm and timing patterns
- **Relevance to Cognitive Impairment**: May indicate speech fluency and motor control; segment statistics changes indicate speech disorders
- **Specific Features**:
  - `loudnessPeaksPerSec`: Number of loudness peaks per second
  - `VoicedSegmentsPerSec`: Number of voiced segments per second
  - `MeanVoicedSegmentLengthSec`: Average length of voiced segments
  - `StddevVoicedSegmentLengthSec`: Standard deviation of voiced segment length
  - `MeanUnvoicedSegmentLength`: Average length of unvoiced segments
  - `StddevUnvoicedSegmentLength`: Standard deviation of unvoiced segment length
  - `equivalentSoundLevel_dBp`: Equivalent sound level in dB

### 3. Fluency & Temporal Features

Fluency and temporal features quantify the efficiency of verbal output, psychomotor speed, language planning, and lexical retrieval—all sensitive to cognitive load and executive dysfunction.

#### Speech Rate (`speech_rate`)
- **Definition**: Syllables or words per second calculated excluding pause intervals
- **Formula**: $SR = \frac{N_{syllables}}{T_{speaking}}$ or $\frac{N_{words}}{T_{total}}$ (words per second)
- **Physiological Significance**: Represents efficiency of verbal output and psychomotor speed
- **Relevance to Cognitive Impairment**: Speech rate decreased in dementia and Parkinson's disease; reflects cognitive slowing

#### Phonation Ratio (`phonation_ratio`)
- **Definition**: Ratio of active speech time to total recording duration
- **Formula**: $PR = \frac{T_{voiced/speech}}{T_{total}}$ where $T_{voiced/speech}$ excludes pauses >200 ms
- **Physiological Significance**: Indicates the density of speech content and sustained attention capacity
- **Relevance to Cognitive Impairment**: Low phonation ratio may indicate increased pausing, word-finding difficulty, or reduced sustained attention; sensitive to cognitive load

#### Articulation Rate (`articulation_rate`)
- **Definition**: Phoneme production rate measured during active speech periods (excluding pauses)
- **Formula**: $AR = \frac{N_{phonemes/syllables}}{T_{active\_speech}}$
- **Physiological Significance**: Reflects pure motor execution capacity independent of cognitive pausing
- **Relevance to Cognitive Impairment**: Articulation rate changes indicate motor speech disorders; distinguishes motor vs. cognitive slowing

#### Hesitation Ratio (`hesitation_ratio`)
- **Definition**: Frequency of filled pauses (e.g., "uh", "um", "嗯", "呃") relative to total words
- **Formula**: $HR = \frac{N_{filled\_pauses}}{N_{total\_words}}$ or $\frac{N_{filled\_pauses}}{N_{total\_words}} \times 100$
- **Physiological Significance**: Directly linked to word-finding difficulties (anomia) and executive dysfunction
- **Relevance to Cognitive Impairment**: Elevated hesitation ratio indicates lexical retrieval deficits; strong marker for MCI and dementia

#### Pause Duration Features
- **Definition**: Average length and frequency of silent intervals (>200 ms) during speech
- **Formula**: $PD_{avg} = \frac{1}{N}\sum_{i} T_{pause_i}$; $PD_{freq} = \frac{N_{pauses}}{T_{total}}$
- **Physiological Significance**: Reflects language planning delays, lexical retrieval deficits, and cognitive load
- **Relevance to Cognitive Impairment**: Pause duration increased in cognitive disorders; longer pauses indicate planning and retrieval difficulty
- **Specific Features**:
  - `short_pause_frequency`: Frequency of short pauses (0.15-0.4s)
  - `long_pause_frequency`: Frequency of long pauses (0.4-1s)
  - `hesitation_frequency`: Frequency of hesitations (>1s)
  - `total_short_pause_duration`: Total duration of short pauses
  - `total_long_pause_duration`: Total duration of long pauses
  - `total_hesitation_duration`: Total duration of hesitations
  - `average_short_pause_duration`: Average duration of short pauses
  - `average_long_pause_duration`: Average duration of long pauses
  - `average_hesitation_duration`: Average duration of hesitations

##### Pause Ratio Features
- **Definition**: Ratio of pause duration to total speech duration
- **Formula**: $PR = \frac{T_{pause}}{T_{total}}$
- **Physiological Significance**: Reflects speech planning and fluency
- **Relevance to Cognitive Impairment**: Increased pause ratios may indicate cognitive load; pause ratios increased in cognitive disorders
- **Specific Features**:
  - `total_short_pause_duration_to_total_speech_duration`: Short pause ratio
  - `total_long_pause_duration_to_total_speech_duration`: Long pause ratio
  - `total_hesitation_duration_to_total_speech_duration`: Hesitation ratio
  - `silence_percentage`: Total silence percentage

#### Speech Segment Features

##### Speech Segment Statistics
- **Definition**: Statistics of speech segments (segments between pauses)
- **Physiological Significance**: Reflects speech fluency and organization
- **Relevance to Cognitive Impairment**: May indicate cognitive processing efficiency; speech segment changes indicate speech disorders
- **Specific Features**:
  - `speech_segment_count`: Number of speech segments
  - `avg_speech_duration`: Average speech segment duration
  - `max_speech_duration`: Maximum speech segment duration
  - `min_speech_duration`: Minimum speech segment duration

##### Phrase Segment Statistics
- **Definition**: Statistics of phrase segments (segments between hesitations)
- **Physiological Significance**: Reflects higher-level speech organization
- **Relevance to Cognitive Impairment**: May indicate cognitive planning and organization; phrase segment changes indicate cognitive disorders
- **Specific Features**:
  - `phrase_segment_count`: Number of phrase segments
  - `avg_phrase_duration`: Average phrase segment duration
  - `max_phrase_duration`: Maximum phrase segment duration
  - `min_phrase_duration`: Minimum phrase segment duration

##### Silence Statistics
- **Definition**: Overall silence and speaking time statistics
- **Physiological Significance**: Reflects overall speech efficiency
- **Relevance to Cognitive Impairment**: May indicate cognitive processing efficiency; silence statistics changes indicate speech disorders
- **Specific Features**:
  - `total_silence_duration`: Total silence duration
  - `total_speech_duration`: Total speaking duration
  - `silence_average_duration`: Average silence duration

#### Rhythm and Timing Features

##### Rhythm Regularity (`rhythm_regularity`)
- **Definition**: Consistency of inter-syllable intervals
- **Formula**: $RR = 1 - \frac{\sigma_{ISI}}{\mu_{ISI}}$ where ISI is inter-syllable interval
- **Physiological Significance**: Reflects motor control and timing
- **Relevance to Cognitive Impairment**: May indicate cognitive motor control; rhythm regularity decreased in Parkinson's disease

##### Syllable Duration Variability (`syllable_duration_var`)
- **Definition**: Variability in syllable durations
- **Formula**: $SDV = \frac{\sigma_{syllable\_duration}}{\mu_{syllable\_duration}}$
- **Physiological Significance**: Reflects motor control and timing precision
- **Relevance to Cognitive Impairment**: May indicate cognitive motor control; syllable duration variability increased in cognitive disorders

##### Stress Pattern (`stress_pattern`)
- **Definition**: Pattern of stressed and unstressed syllables
- **Formula**: $SP = \frac{N_{stressed}}{N_{total}}$
- **Physiological Significance**: Reflects prosodic control and expression
- **Relevance to Cognitive Impairment**: May indicate emotional expression and communication; stress pattern changes indicate prosodic disorders

### 4. Voice Quality Features

Voice quality features capture cycle-to-cycle stability of vocal fold vibration, reflecting neuromotor control and glottal function—sensitive to lack of control and breathiness linked to vagal tone.

#### Jitter (`jitter`, `jitterLocal_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in fundamental frequency (pitch perturbation)
- **Formula**: $Jitter = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|F_0(i+1) - F_0(i)|}{F_0(i)}$ (relative or absolute)
- **Physiological Significance**: Sensitive to lack of control over vocal fold vibration (neuromotor instability)
- **Relevance to Cognitive Impairment**: Jitter increased in neurological disorders, depression, and Parkinson's disease

#### Shimmer (`shimmer`, `shimmerLocaldB_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in amplitude (loudness perturbation)
- **Formula**: $Shimmer = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|A(i+1) - A(i)|}{A(i)}$ (in dB: $20\log_{10}(A_{i+1}/A_i)$)
- **Physiological Significance**: Reflects reduced glottal closure and breathiness potentially linked to vagal tone
- **Relevance to Cognitive Impairment**: Shimmer increased in neurological disorders and depression

#### Harmonics-to-Noise Ratio (`HNR`, `HNRdBACF_sma3nz_*`)
- **Definition**: Harmonics-to-Noise Ratio—periodic vs. aperiodic energy in voiced segments
- **Formula**: $HNR = 10\log_{10}\left(\frac{P_{harmonic}}{P_{noise}}\right)$ (dB)
- **Physiological Significance**: Quantifies hoarseness or breathiness; indicates degradation in voice purity
- **Relevance to Cognitive Impairment**: HNR decreased in voice disorders, depression, and cognitive impairment

##### Glottal-to-Noise Excitation Ratio (`GNE`)
- **Definition**: Ratio of glottal to noise excitation
- **Formula**: $GNE = \frac{P_{glottal}}{P_{noise}}$
- **Physiological Significance**: Reflects voice source characteristics
- **Relevance to Cognitive Impairment**: May indicate voice quality; GNE changes indicate voice quality variations

#### Emotional and Affective Features

##### Emotional Valence (`emotional_valence`)
- **Definition**: Positive or negative emotional content
- **Formula**: $EV = \frac{1}{N}\sum_{i=1}^{N} valence_i$
- **Physiological Significance**: Reflects emotional state and expression
- **Relevance to Cognitive Impairment**: May indicate mood and emotional regulation; emotional valence changes indicate mood disorders

##### Emotional Arousal (`emotional_arousal`)
- **Definition**: Level of emotional activation
- **Formula**: $EA = \frac{1}{N}\sum_{i=1}^{N} arousal_i$
- **Physiological Significance**: Reflects emotional intensity
- **Relevance to Cognitive Impairment**: May indicate emotional regulation; emotional arousal changes indicate emotional disorders

##### Emotional Dominance (`emotional_dominance`)
- **Definition**: Level of emotional control
- **Formula**: $ED = \frac{1}{N}\sum_{i=1}^{N} dominance_i$
- **Physiological Significance**: Reflects emotional control and regulation
- **Relevance to Cognitive Impairment**: May indicate emotional regulation; emotional dominance changes indicate emotional disorders

### 5. Semantic Features

Semantic features capture the linguistic content, coherence, and structural complexity of speech—reflecting deep semantic processing, memory retrieval, and grammatical ability altered in cognitive impairment.

#### BERT Embeddings / Match Score (`Match Score`, `BERT_emb`)
- **Definition**: Contextual semantic vectors from transformer-based language models; cosine similarity between speech and reference
- **Formula**: $MatchScore = \frac{e_1 \cdot e_2}{||e_1|| \cdot ||e_2||}$ where $e_1, e_2 \in \mathbb{R}^{768}$ (BERT embeddings)
- **Physiological Significance**: Captures deep coherence and contextual relevance of spoken narratives
- **Relevance to Cognitive Impairment**: Match score decreased in cognitive disorders; indicates semantic memory and processing ability
- **Implementation**: BERT-Chinese model generates 768-dimensional sentence embeddings

#### Semantic Similarity (`semantic_similarity`)
- **Definition**: Cosine distance or similarity between speech content and target references
- **Formula**: $Sim = \cos(\theta) = \frac{e_{speech} \cdot e_{ref}}{||e_{speech}|| \cdot ||e_{ref}||}$
- **Physiological Significance**: Indicates alignment of expression with target meaning, reflecting comprehension
- **Relevance to Cognitive Impairment**: Reduced similarity indicates semantic and comprehension deficits in dementia

#### Coverage Score / Semantic Coverage (`Coverage Score`)
- **Definition**: Proportion of key narrative elements (Information Units) expressed
- **Formula**: $CoverageScore = \frac{N_{covered\_elements}}{N_{total\_elements}}$
- **Physiological Significance**: Reflects completeness of information encoding and memory retrieval
- **Relevance to Cognitive Impairment**: Coverage score decreased in cognitive disorders; indicates attention, memory, and semantic processing
- **Key Elements**: 19 semantic units including characters (男孩, 女孩, 女人), locations (家, 窗外), objects (饼干, 饼干盒, 凳子, 水池, 水龙头, 盘子, 毛巾, 桌子), and actions (站, 拿, 倒, 扶, 洗盘子, 水流出来)

#### Type-Token Ratio (`TTR`)
- **Definition**: Ratio of unique words (types) to total words (tokens)
- **Formula**: $TTR = \frac{N_{types}}{N_{tokens}}$
- **Physiological Significance**: Classic measure of lexical diversity
- **Relevance to Cognitive Impairment**: Low TTR indicates repetitive speech; decreased in dementia and MCI; reflects lexical access and cognitive reserve

#### Syntax Depth (`syntax_depth`)
- **Definition**: Depth of dependency parse trees or mean clause length
- **Formula**: $SD = \max_{nodes} depth(tree)$ or $SD = \frac{1}{N}\sum length(clause_i)$
- **Physiological Significance**: Indicates the ability to construct complex grammatical structures
- **Relevance to Cognitive Impairment**: Reduced syntax depth in dementia; reflects grammatical processing and executive function

<!-- 
##### Lexical Density (`lexical_density`)
- **Definition**: Proportion of content words
- **Formula**: $LD = \frac{N_{content\_words}}{N_{total\_words}}$
- **Physiological Significance**: Reflects semantic content density
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Lexical density changes indicate cognitive disorders

##### Word Length (`word_length`)
- **Definition**: Average length of words used
- **Formula**: $WL = \frac{1}{N}\sum_{i=1}^{N} length(word_i)$
- **Physiological Significance**: Reflects lexical complexity
- **Cognitive Relevance**: May indicate cognitive complexity
- **Clinical Application**: Word length changes indicate cognitive disorders -->

<!-- 
#### 3.2 Syntactic Features

##### Mean Length of Utterance (`MLU`)
- **Definition**: Average number of words per utterance
- **Formula**: $MLU = \frac{1}{N}\sum_{i=1}^{N} length(utterance_i)$
- **Physiological Significance**: Reflects syntactic complexity
- **Cognitive Relevance**: May indicate cognitive complexity and processing
- **Clinical Application**: MLU is decreased in cognitive disorders

##### Syntactic Complexity (`syntactic_complexity`)
- **Definition**: Complexity of syntactic structures
- **Formula**: $SC = \frac{N_{complex\_structures}}{N_{total\_structures}}$
- **Physiological Significance**: Reflects syntactic processing ability
- **Cognitive Relevance**: May indicate cognitive complexity
- **Clinical Application**: Syntactic complexity is decreased in cognitive disorders

##### Grammar Errors (`grammar_errors`)
- **Definition**: Number of grammatical errors
- **Formula**: $GE = \sum_{i=1}^{N} error_i$
- **Physiological Significance**: Reflects grammatical knowledge and control
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Grammar errors are increased in cognitive disorders

##### Sentence Complexity (`sentence_complexity`)
- **Definition**: Complexity of sentence structures
- **Formula**: $SC = \frac{1}{N}\sum_{i=1}^{N} complexity(sentence_i)$
- **Physiological Significance**: Reflects syntactic processing ability
- **Cognitive Relevance**: May indicate cognitive complexity
- **Clinical Application**: Sentence complexity is decreased in cognitive disorders

#### 3.3 Discourse Features

##### Coherence (`coherence`)
- **Definition**: Logical flow and organization of speech
- **Formula**: $C = \frac{1}{N}\sum_{i=1}^{N} coherence\_score(segment_i)$
- **Physiological Significance**: Reflects discourse planning and organization
- **Cognitive Relevance**: May indicate cognitive organization and planning
- **Clinical Application**: Coherence is decreased in cognitive disorders

##### Topic Maintenance (`topic_maintenance`)
- **Definition**: Ability to maintain topic focus
- **Formula**: $TM = \frac{N_{topic\_consistent\_utterances}}{N_{total\_utterances}}$
- **Physiological Significance**: Reflects attention and focus
- **Cognitive Relevance**: May indicate attention and executive function
- **Clinical Application**: Topic maintenance is decreased in cognitive disorders

##### Discourse Markers (`discourse_markers`)
- **Definition**: Use of discourse markers and connectors
- **Formula**: $DM = \frac{N_{discourse\_markers}}{N_{total\_words}}$
- **Physiological Significance**: Reflects discourse organization
- **Cognitive Relevance**: May indicate cognitive organization
- **Clinical Application**: Discourse markers are decreased in cognitive disorders

##### Repetition (`repetition`)
- **Definition**: Frequency of repeated words or phrases
- **Formula**: $R = \frac{N_{repetitions}}{N_{total\_words}}$
- **Physiological Significance**: Reflects memory and retrieval processes
- **Cognitive Relevance**: May indicate memory difficulties
- **Clinical Application**: Repetition is increased in cognitive disorders

#### 3.4 Content Analysis Features

##### Semantic Similarity (`semantic_similarity`)
- **Definition**: Similarity between words and concepts
- **Formula**: $SS = \frac{1}{N}\sum_{i=1}^{N} similarity(word_i, context_i)$
- **Physiological Significance**: Reflects semantic knowledge and processing
- **Cognitive Relevance**: May indicate semantic memory and processing
- **Clinical Application**: Semantic similarity changes indicate semantic memory disorders

##### Concept Density (`concept_density`)
- **Definition**: Density of concepts per unit of speech
- **Formula**: $CD = \frac{N_{concepts}}{N_{total\_words}}$
- **Physiological Significance**: Reflects conceptual processing
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Concept density changes indicate cognitive disorders

##### Information Density (`information_density`)
- **Definition**: Amount of information per unit of speech
- **Formula**: $ID = \frac{information\_content}{speech\_duration}$
- **Physiological Significance**: Reflects cognitive processing efficiency
- **Cognitive Relevance**: May indicate cognitive efficiency
- **Clinical Application**: Information density is decreased in cognitive disorders

##### Semantic Coherence (`semantic_coherence`)
- **Definition**: Semantic consistency within discourse
- **Formula**: $SC = \frac{1}{N}\sum_{i=1}^{N} semantic\_coherence(segment_i)$
- **Physiological Significance**: Reflects semantic processing and organization
- **Cognitive Relevance**: May indicate semantic memory and processing
- **Clinical Application**: Semantic coherence is decreased in cognitive disorders -->

## Implementation Pipeline

### Data Preprocessing
1. **Audio Preprocessing**:
   - Sampling rate standardization to 16 kHz for frequency consistency
   - Frame windowing and noise reduction
   - Voice activity detection (VAD) using RMS energy threshold (0.1)
   - Audio segmentation and normalization

2. **Text Preprocessing**:
   - Automatic speech recognition using WeNet end-to-end framework
   - Chinese speech recognition with pre-trained models
   - Text normalization and cleaning
   - BERT tokenization using BertTokenizer

### Feature Extraction Pipeline
1. **Acoustic Feature Extraction**:
   - Use OpenSMILE toolkit with eGeMAPSV02 feature set
   - Extract 88-dimensional acoustic features including F0, loudness, MFCC, formants, spectral features
   - Compute statistical functionals (mean, std, percentiles, slopes) from low-level descriptors

2. **Fluency & Temporal Feature Extraction**:
   - Analyze pause patterns using energy threshold (0.1)
   - Categorize pauses: short (0.15-0.4s), long (0.4-1s), hesitation (>1s)
   - Compute speech rate (syllables/words per second excluding pauses), articulation rate, phonation ratio
   - Detect filled pauses (uh, um, 嗯, 呃) for hesitation ratio
   - Compute pause frequencies, durations, segment statistics

3. **Semantic Feature Extraction**:
   - Use WeNet ASR for speech-to-text conversion
   - Apply BERT-Chinese model for semantic analysis
   - Compute Match Score (cosine similarity), Coverage Score (key element coverage)
   - Compute Type-Token Ratio (lexical diversity) and Syntax Depth (dependency parse or clause length)

### Quality Control Measures
- **Audio Quality**: Ensure adequate signal-to-noise ratio (>20 dB)
- **Text Quality**: Verify ASR accuracy and text quality
- **Feature Validation**: Check feature ranges and distributions
- **Missing Data**: Handle gaps and artifacts appropriately

<!-- ## Clinical Significance and Applications

### Cognitive Assessment Biomarkers
Speech features provide objective measures of cognitive function:

- **Language Processing**: Vocabulary, syntax, and semantic features
- **Motor Control**: Articulation, prosody, and voice quality features
- **Executive Function**: Discourse organization and topic maintenance
- **Memory**: Repetition and semantic memory features

### Disease-Specific Applications

#### Alzheimer's Disease
- **Reduced Vocabulary**: Decreased vocabulary richness and TTR
- **Simplified Syntax**: Reduced MLU and syntactic complexity
- **Discourse Impairment**: Decreased coherence and topic maintenance
- **Semantic Deficits**: Reduced semantic similarity and coherence

#### Mild Cognitive Impairment (MCI)
- **Subtle Language Changes**: Early changes in vocabulary and syntax
- **Discourse Difficulties**: Mild impairments in coherence and organization
- **Semantic Processing**: Subtle changes in semantic features
- **Prosodic Changes**: Early changes in prosodic features

#### Parkinson's Disease
- **Voice Quality**: Changes in jitter, shimmer, and HNR
- **Prosodic Impairment**: Reduced F0 variability and prosodic features
- **Speech Rate**: Slowed speech rate and articulation
- **Rhythm Changes**: Altered rhythm and timing features

#### Frontotemporal Dementia
- **Language Impairment**: Significant vocabulary and syntax changes
- **Discourse Deficits**: Severe impairments in coherence and organization
- **Semantic Deficits**: Major changes in semantic processing
- **Behavioral Changes**: Changes in emotional and affective features

### Early Detection and Monitoring
Speech features may detect cognitive changes before behavioral symptoms:

- **Language Changes**: Early vocabulary and syntax changes
- **Prosodic Changes**: Early prosodic and voice quality changes
- **Discourse Changes**: Early discourse organization changes
- **Semantic Changes**: Early semantic processing changes

## Technical Considerations

### Signal Processing Requirements
- **Computational Complexity**: Some features require significant computation
- **Memory Requirements**: Large datasets require efficient processing
- **Real-time Processing**: Consider computational constraints for clinical applications
- **Robustness**: Features should be robust to noise and artifacts

### Statistical Considerations
- **Multiple Comparisons**: Correct for multiple testing when using many features
- **Effect Sizes**: Consider practical significance beyond statistical significance
- **Longitudinal Analysis**: Account for within-subject correlations
- **Cross-validation**: Ensure robust model performance

### Clinical Validation
- **Population Norms**: Establish normal ranges for different populations
- **Age and Gender Effects**: Account for demographic factors
- **Language Effects**: Consider language-specific characteristics
- **Comorbidities**: Consider multiple health conditions

## References

### Key Literature

1. **Speech and Cognitive Assessment**
   - Fraser, K. C., et al. (2016). Linguistic features identify Alzheimer's disease in narrative speech. *Journal of Alzheimer's Disease*, 49(2), 407-422.
   - König, A., et al. (2018). Automatic speech analysis for the assessment of patients with predementia and Alzheimer's disease. *Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring*, 10, 561-578.

2. **Acoustic Analysis**
   - Quatieri, T. F. (2002). *Discrete-time speech signal processing: principles and practice*. Prentice Hall.
   - Rabiner, L. R., & Schafer, R. W. (2011). *Theory and applications of digital speech processing*. Pearson.

3. **Paralinguistic Features**
   - Schuller, B., et al. (2013). The INTERSPEECH 2013 computational paralinguistics challenge: Social signals, conflict, emotion, autism. *Proceedings of INTERSPEECH*, 148-152.
   - Eyben, F., et al. (2010). Opensmile: the munich versatile and fast open-source audio feature extractor. *Proceedings of the 18th ACM international conference on Multimedia*, 1459-1462.

4. **Semantic Analysis**
   - Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.
   - Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

5. **Clinical Applications**
   - Roark, B., et al. (2011). Spoken language derived measures for detecting mild cognitive impairment. *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7), 2081-2090.
   - López-de-Ipiña, K., et al. (2013). On the selection of non-invasive methods based on speech analysis oriented to automatic Alzheimer disease diagnosis. *Sensors*, 13(5), 6730-6745.

6. **Speech Processing for Cognitive Assessment**
   - Weiner, J., et al. (2016). Automatic detection of cognitive impairment in elderly people using an entertainment social robot. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 24(12), 1375-1386.
   - Tóth, L., et al. (2018). A speech recognition-based solution for the automatic detection of mild cognitive impairment from spontaneous speech. *Current Alzheimer Research*, 15(2), 130-138. -->

### Open-Source Code Libraries and Tools

#### Python Libraries

1. **OpenSMILE**: Open-source audio feature extractor
   - Repository: https://github.com/audeering/opensmile-python
   - Website: https://www.audeering.com/opensmile/
   - Features: Comprehensive audio feature extraction, emotion recognition, paralinguistic analysis
   - Documentation: https://audeering.github.io/opensmile-python/

2. **Librosa**: Audio and music signal processing
   - Repository: https://github.com/librosa/librosa
   - Website: https://librosa.org/
   - Features: Audio feature extraction, spectral analysis, tempo estimation
   - Documentation: https://librosa.org/doc/latest/

3. **SpeechRecognition**: Speech recognition library
   - Repository: https://github.com/Uberi/speech_recognition
   - Website: https://pypi.org/project/SpeechRecognition/
   - Features: Multiple ASR engines, audio preprocessing
   - Documentation: https://pypi.org/project/SpeechRecognition/

4. **Transformers (HuggingFace)**: Pre-trained language models
   - Repository: https://github.com/huggingface/transformers
   - Website: https://huggingface.co/transformers/
   - Features: BERT, RoBERTa, GPT models, multilingual support
   - Documentation: https://huggingface.co/docs/transformers/

5. **BERT-Chinese Models**:
   - **Chinese BERT**: https://huggingface.co/bert-base-chinese
   - **Chinese RoBERTa**: https://huggingface.co/hfl/chinese-roberta-wwm-ext
   - **Chinese GPT**: https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
   - Features: Pre-trained Chinese language models for semantic analysis

6. **SpaCy**: Industrial-strength natural language processing
   - Repository: https://github.com/explosion/spaCy
   - Website: https://spacy.io/
   - Features: NLP pipeline, named entity recognition, dependency parsing
   - Documentation: https://spacy.io/usage

7. **NLTK**: Natural Language Toolkit
   - Repository: https://github.com/nltk/nltk
   - Website: https://www.nltk.org/
   - Features: Text processing, tokenization, POS tagging, semantic analysis
   - Documentation: https://www.nltk.org/

8. **PyAudioAnalysis**: Audio feature extraction and classification
   - Repository: https://github.com/tyiannak/pyAudioAnalysis
   - Website: https://github.com/tyiannak/pyAudioAnalysis
   - Features: Audio feature extraction, classification, segmentation
   - Documentation: https://github.com/tyiannak/pyAudioAnalysis


<!-- 
#### R Libraries

1. **tuneR**: Audio analysis and music information retrieval
   - Repository: https://cran.r-project.org/web/packages/tuneR/
   - Features: Audio file handling, spectral analysis, pitch detection
   - Documentation: https://cran.r-project.org/web/packages/tuneR/

2. **seewave**: Sound analysis and synthesis
   - Repository: https://cran.r-project.org/web/packages/seewave/
   - Features: Audio analysis, spectral features, acoustic measurements
   - Documentation: https://cran.r-project.org/web/packages/seewave/

#### MATLAB Toolboxes

1. **Voicebox**: Speech processing toolbox
   - Website: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   - Features: Speech analysis, feature extraction, voice quality
   - Documentation: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

2. **Praat**: Phonetics software
   - Website: https://www.fon.hum.uva.nl/praat/
   - Features: Speech analysis, phonetics, prosody
   - Documentation: https://www.fon.hum.uva.nl/praat/ -->
   

#### Online Resources and Tutorials

1. **OpenSMILE Tutorial**: https://www.audeering.com/opensmile/
2. **HuggingFace Transformers Tutorial**: https://huggingface.co/course
3. **Librosa Tutorial**: https://librosa.org/doc/latest/tutorial.html
4. **Speech Recognition Tutorial**: https://realpython.com/python-speech-recognition/
5. **Chinese NLP with BERT**: https://github.com/ymcui/Chinese-BERT-wwm

### GitHub Speech Processing Projects

1. **WeNet**: End-to-end speech recognition framework
   - Repository: https://github.com/wenet-e2e/wenet
   - Website: https://github.com/wenet-e2e/wenet
   - Features: End-to-end ASR, Chinese speech recognition, pre-trained models
   - Documentation: https://github.com/wenet-e2e/wenet

2. **Chinese BERT Models**:
   - **Chinese BERT**: https://github.com/ymcui/Chinese-BERT-wwm
   - **Chinese RoBERTa**: https://github.com/ymcui/Chinese-RoBERTa-wwm-ext
   - Features: Pre-trained Chinese language models, semantic analysis
   - Documentation: https://github.com/ymcui/Chinese-BERT-wwm

3. **Speech Recognition**: https://github.com/Uberi/speech_recognition
4. **Audio Feature Extraction**: https://github.com/tyiannak/pyAudioAnalysis
5. **Speech Emotion Recognition**: https://github.com/topics/speech-emotion-recognition
6. **Voice Activity Detection**: https://github.com/wiseman/py-webrtcvad

### Key Features for MCI Detection

Based on correlation analysis with cognitive outcomes, the most important features for MCI detection are (in order of correlation strength):

1. `loudness_sma3_stddevNorm` - Loudness variability
2. `Match Score` - Semantic similarity with reference
3. `Coverage Score` - Coverage of key semantic elements
4. `hammarbergIndexUV_sma3nz_amean` - Hammarberg index for unvoiced segments
5. `F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope` - F0 rising slope
6. `Phrase Segment Avg Duration (Sec)` - Average phrase duration
7. `spectralFluxV_sma3nz_stddevNorm` - Spectral flux variability for voiced segments
8. `mfcc1_sma3_stddevNorm` - MFCC1 variability
9. `Phrase Segment Max Duration (Sec)` - Maximum phrase duration
10. `slopeV0-500_sma3nz_amean` - Spectral slope in 0-500Hz range

## Conclusion

The comprehensive speech feature set described in this document provides a robust foundation for cognitive impairment assessment through speech analysis. The features span five major categories aligned with the expanded speech taxonomy:

1. **Spectral features**: MFCC (1–13), Formants (F1–F3), Spectral Centroid, Spectral Flux—capturing vocal tract characteristics and articulatory precision.
2. **Prosodic features**: F0 (Mean/Std), Intensity, Pitch Contour—reflecting vocal control, intonation, and affective modulation.
3. **Fluency & Temporal**: Speech Rate, Pause Duration, Phonation Ratio, Articulation Rate, Hesitation Ratio—quantifying verbal output efficiency and lexical retrieval.
4. **Voice Quality**: Jitter, Shimmer, HNR—capturing neuromotor stability and glottal function.
5. **Semantic features**: BERT Embeddings, Semantic Similarity, Semantic Coverage, Type-Token Ratio, Syntax Depth—reflecting coherence, lexical diversity, and grammatical complexity.

When combined with appropriate signal processing techniques and clinical validation, these features can serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders. The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of speech production and content.

The extensive literature support and open-source code libraries, including OpenSMILE for acoustic analysis and BERT-Chinese for semantic analysis, provide researchers and clinicians with the tools necessary to implement and validate these features in their own studies, contributing to the advancement of cognitive health assessment through speech analysis.
