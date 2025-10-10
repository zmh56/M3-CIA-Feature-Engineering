# Speech Feature Descriptions for Cognitive Impairment Assessment

This document provides comprehensive descriptions of speech features extracted for cognitive impairment assessment in the M3-CIA framework. Speech features are extracted from picture description tasks (specifically the Cookie Theft picture from the Boston Diagnostic Aphasia Examination) and categorized into three main domains: acoustic features (88 dimensions), paralinguistic features (25 dimensions), and semantic features (2 dimensions), totaling 115 dimensions that capture various aspects of speech production and content closely linked to cognitive performance and brain health.

## Overview

Speech features are extracted from the Cookie Theft picture description task, a standardized cognitive assessment tool that requires participants to describe a complex scene. This task captures the acoustic properties, paralinguistic characteristics, and semantic content of speech production under cognitive load. These features serve as objective biomarkers for detecting and monitoring cognitive impairment through speech analysis, providing non-invasive assessment of cognitive function.

## Feature Categories and Descriptions

### 1. Acoustic Features (88 dimensions)

Acoustic features capture the fundamental physical properties of speech signals, including frequency, amplitude, and temporal characteristics. These features are extracted using the OpenSMILE toolkit with the eGeMAPSV02 feature set, which includes Low-Level Descriptors (LLD) and Functionals to capture prosody, voice quality, and spectral information.

#### 1.1 Fundamental Frequency (F0) Features

##### F0 Semitone Features (`F0semitoneFrom27.5Hz_sma3nz_*`)
- **Definition**: Fundamental frequency converted to semitone scale from 27.5Hz reference
- **Formula**: $F_0^{semitone} = 12 \times \log_2\left(\frac{F_0}{27.5}\right)$
- **Physiological Significance**: Reflects vocal cord vibration rate and laryngeal function
- **Cognitive Relevance**: May indicate emotional state, stress, or neurological changes
- **Clinical Application**: F0 changes are associated with Parkinson's disease and depression
- **Specific Features**:
  - `F0semitoneFrom27.5Hz_sma3nz_amean`: Mean F0 in semitone scale
  - `F0semitoneFrom27.5Hz_sma3nz_stddevNorm`: Normalized standard deviation
  - `F0semitoneFrom27.5Hz_sma3nz_percentile20.0/50.0/80.0`: Percentile values
  - `F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope/FallingSlope`: Slope features

##### Fundamental Frequency Standard Deviation (`F0_std`)
- **Definition**: Variability of fundamental frequency
- **Formula**: $F_0^{std} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(F_0(i) - F_0^{mean})^2}$
- **Physiological Significance**: Indicates vocal control and stability
- **Cognitive Relevance**: Reduced variability may indicate cognitive decline
- **Clinical Application**: F0 variability is decreased in Alzheimer's disease

##### Fundamental Frequency Range (`F0_range`)
- **Definition**: Difference between maximum and minimum F0 values
- **Formula**: $F_0^{range} = F_0^{max} - F_0^{min}$
- **Physiological Significance**: Reflects vocal range and flexibility
- **Cognitive Relevance**: May indicate cognitive flexibility and expression
- **Clinical Application**: F0 range is reduced in cognitive disorders

##### Fundamental Frequency Slope (`F0_slope`)
- **Definition**: Rate of change of F0 over time
- **Formula**: $F_0^{slope} = \frac{dF_0}{dt}$
- **Physiological Significance**: Indicates intonation patterns and prosody
- **Cognitive Relevance**: May reflect emotional expression and communication
- **Clinical Application**: F0 slope changes indicate prosodic deficits

#### 1.2 Formant Features

##### First Formant (`F1`)
- **Definition**: First formant frequency
- **Formula**: $F1 = \text{first resonance frequency of vocal tract}$
- **Physiological Significance**: Reflects tongue height and jaw position
- **Cognitive Relevance**: May indicate articulation precision
- **Clinical Application**: F1 changes indicate articulation disorders

##### Second Formant (`F2`)
- **Definition**: Second formant frequency
- **Formula**: $F2 = \text{second resonance frequency of vocal tract}$
- **Physiological Significance**: Reflects tongue front-back position
- **Cognitive Relevance**: May indicate speech clarity and precision
- **Clinical Application**: F2 changes indicate articulation and vowel production

##### Third Formant (`F3`)
- **Definition**: Third formant frequency
- **Formula**: $F3 = \text{third resonance frequency of vocal tract}$
- **Physiological Significance**: Reflects tongue tip position and lip rounding
- **Cognitive Relevance**: May indicate speech precision and clarity
- **Clinical Application**: F3 changes indicate articulation disorders

##### Formant Bandwidth (`F1_BW`, `F2_BW`, `F3_BW`)
- **Definition**: Bandwidth of formant frequencies
- **Formula**: $F_i^{BW} = \text{bandwidth of } F_i$
- **Physiological Significance**: Reflects vocal tract damping
- **Cognitive Relevance**: May indicate speech quality and clarity
- **Clinical Application**: Formant bandwidth changes indicate speech quality degradation

#### 1.3 Loudness Features

##### Loudness Features (`loudness_sma3_*`)
- **Definition**: Perceptual loudness measures using spectral analysis
- **Formula**: $Loudness = \sqrt[0.67]{\sum_{k=1}^{K} |X(k)|^{0.67}}$
- **Physiological Significance**: Reflects perceived vocal effort and loudness
- **Cognitive Relevance**: May indicate speech effort and energy
- **Clinical Application**: Loudness changes indicate vocal effort variations
- **Specific Features**:
  - `loudness_sma3_amean`: Mean loudness
  - `loudness_sma3_stddevNorm`: Normalized standard deviation
  - `loudness_sma3_percentile20.0/50.0/80.0`: Percentile values
  - `loudness_sma3_meanRisingSlope/FallingSlope`: Slope features

##### Zero Crossing Rate (`ZCR`)
- **Definition**: Rate of sign changes in the signal
- **Formula**: $ZCR = \frac{1}{N-1}\sum_{i=1}^{N-1} |sgn(x(i)) - sgn(x(i+1))|$
- **Physiological Significance**: Distinguishes between voiced and unvoiced segments
- **Cognitive Relevance**: May indicate speech rhythm and timing
- **Clinical Application**: ZCR changes indicate voice quality variations

##### Spectral Centroid (`spectral_centroid`)
- **Definition**: Center of mass of the spectrum
- **Formula**: $SC = \frac{\sum_{k=1}^{K} k \cdot |X(k)|}{\sum_{k=1}^{K} |X(k)|}$
- **Physiological Significance**: Reflects brightness of the voice
- **Cognitive Relevance**: May indicate emotional state and energy
- **Clinical Application**: Spectral centroid changes indicate voice quality

##### Spectral Rolloff (`spectral_rolloff`)
- **Definition**: Frequency below which 85% of spectral energy is contained
- **Formula**: $SR = \min(f) \text{ such that } \sum_{k=1}^{f} |X(k)| = 0.85 \sum_{k=1}^{K} |X(k)|$
- **Physiological Significance**: Reflects spectral shape and voice quality
- **Cognitive Relevance**: May indicate speech clarity and quality
- **Clinical Application**: Spectral rolloff changes indicate voice quality degradation

#### 1.4 Mel-Frequency Cepstral Coefficients (MFCC)

##### MFCC Features (`mfcc1-4_sma3_*`)
- **Definition**: Mel-frequency cepstral coefficients capturing spectral envelope
- **Formula**: $MFCC_i = \sum_{k=1}^{K} \log(|X(k)|) \cos\left(\frac{i(k-0.5)\pi}{K}\right)$
- **Physiological Significance**: Captures spectral envelope characteristics
- **Cognitive Relevance**: May indicate speech clarity and articulation
- **Clinical Application**: MFCC changes indicate articulation disorders
- **Specific Features**:
  - `mfcc1-4_sma3_amean`: Mean values of first 4 MFCC coefficients
  - `mfcc1-4_sma3_stddevNorm`: Normalized standard deviations
  - `mfcc1-4V_sma3nz_amean/stddevNorm`: MFCC for voiced segments

##### Spectral Bandwidth (`spectral_bandwidth`)
- **Definition**: Width of the spectrum around the centroid
- **Formula**: $SB = \sqrt{\frac{\sum_{k=1}^{K} (k - SC)^2 \cdot |X(k)|}{\sum_{k=1}^{K} |X(k)|}}$
- **Physiological Significance**: Reflects spectral spread and voice quality
- **Cognitive Relevance**: May indicate speech clarity and quality
- **Clinical Application**: Spectral bandwidth changes indicate voice quality variations

##### Spectral Contrast (`spectral_contrast`)
- **Definition**: Difference between peak and valley in spectral sub-bands
- **Formula**: $SC = \frac{1}{M}\sum_{m=1}^{M} \log\left(\frac{P_m}{V_m}\right)$
- **Physiological Significance**: Reflects spectral dynamics and timbre
- **Cognitive Relevance**: May indicate speech expressiveness
- **Clinical Application**: Spectral contrast changes indicate voice quality

#### 1.5 Voice Quality Features

##### Jitter Features (`jitterLocal_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in fundamental frequency
- **Formula**: $Jitter = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|F_0(i+1) - F_0(i)|}{F_0(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Cognitive Relevance**: May indicate neurological control
- **Clinical Application**: Jitter is increased in neurological disorders
- **Specific Features**: `jitterLocal_sma3nz_amean`, `jitterLocal_sma3nz_stddevNorm`

##### Shimmer Features (`shimmerLocaldB_sma3nz_*`)
- **Definition**: Cycle-to-cycle variation in amplitude (in dB)
- **Formula**: $Shimmer = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|A(i+1) - A(i)|}{A(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Cognitive Relevance**: May indicate neurological control
- **Clinical Application**: Shimmer is increased in neurological disorders
- **Specific Features**: `shimmerLocaldB_sma3nz_amean`, `shimmerLocaldB_sma3nz_stddevNorm`

##### Harmonics-to-Noise Ratio (`HNRdBACF_sma3nz_*`)
- **Definition**: Ratio of harmonic to noise energy using autocorrelation function
- **Formula**: $HNR = 10\log_{10}\left(\frac{P_{harmonic}}{P_{noise}}\right)$
- **Physiological Significance**: Reflects voice quality and clarity
- **Cognitive Relevance**: May indicate vocal control and health
- **Clinical Application**: HNR is decreased in voice disorders
- **Specific Features**: `HNRdBACF_sma3nz_amean`, `HNRdBACF_sma3nz_stddevNorm`

#### 1.6 Spectral Shape Features

##### Spectral Slope Features (`slopeV0-500_sma3nz_*`, `slopeV500-1500_sma3nz_*`)
- **Definition**: Spectral slope in different frequency ranges for voiced segments
- **Formula**: $Slope = \frac{\log(P_{f2}) - \log(P_{f1})}{\log(f2) - \log(f1)}$
- **Physiological Significance**: Reflects spectral shape and voice quality
- **Cognitive Relevance**: May indicate speech clarity and quality
- **Clinical Application**: Spectral slope changes indicate voice quality variations
- **Specific Features**: 
  - `slopeV0-500_sma3nz_amean/stddevNorm`: Slope in 0-500Hz range
  - `slopeV500-1500_sma3nz_amean/stddevNorm`: Slope in 500-1500Hz range

##### Alpha Ratio (`alphaRatioV_sma3nz_*`)
- **Definition**: Ratio of energy in 50-1000Hz to 1-5kHz for voiced segments
- **Formula**: $AlphaRatio = \frac{E_{50-1000Hz}}{E_{1-5kHz}}$
- **Physiological Significance**: Reflects spectral balance and voice quality
- **Cognitive Relevance**: May indicate speech clarity and quality
- **Clinical Application**: Alpha ratio changes indicate voice quality variations

##### Hammarberg Index (`hammarbergIndexV_sma3nz_*`, `hammarbergIndexUV_sma3nz_*`)
- **Definition**: Ratio of maximum energy in 0-2kHz to maximum energy in 2-5kHz
- **Formula**: $HI = \frac{\max(E_{0-2kHz})}{\max(E_{2-5kHz})}$
- **Physiological Significance**: Reflects spectral balance and voice quality
- **Cognitive Relevance**: May indicate speech clarity and quality
- **Clinical Application**: Hammarberg index changes indicate voice quality variations
- **Specific Features**: 
  - `hammarbergIndexV_sma3nz_amean/stddevNorm`: For voiced segments
  - `hammarbergIndexUV_sma3nz_amean`: For unvoiced segments

#### 1.7 Spectral Flux Features

##### Spectral Flux (`spectralFlux_sma3_*`, `spectralFluxV_sma3nz_*`, `spectralFluxUV_sma3nz_*`)
- **Definition**: Rate of change of spectral energy over time
- **Formula**: $SF = \sum_{k=1}^{K} |X_{t+1}(k) - X_t(k)|$
- **Physiological Significance**: Reflects spectral dynamics and timbre changes
- **Cognitive Relevance**: May indicate speech expressiveness and clarity
- **Clinical Application**: Spectral flux changes indicate voice quality variations
- **Specific Features**: 
  - `spectralFlux_sma3_amean/stddevNorm`: Overall spectral flux
  - `spectralFluxV_sma3nz_amean/stddevNorm`: For voiced segments
  - `spectralFluxUV_sma3nz_amean`: For unvoiced segments

#### 1.8 Segment Statistics Features

##### Voiced/Unvoiced Segment Features
- **Definition**: Statistics of voiced and unvoiced speech segments
- **Physiological Significance**: Reflects speech rhythm and timing patterns
- **Cognitive Relevance**: May indicate speech fluency and motor control
- **Clinical Application**: Segment statistics changes indicate speech disorders
- **Specific Features**:
  - `loudnessPeaksPerSec`: Number of loudness peaks per second
  - `VoicedSegmentsPerSec`: Number of voiced segments per second
  - `MeanVoicedSegmentLengthSec`: Average length of voiced segments
  - `StddevVoicedSegmentLengthSec`: Standard deviation of voiced segment length
  - `MeanUnvoicedSegmentLength`: Average length of unvoiced segments
  - `StddevUnvoicedSegmentLength`: Standard deviation of unvoiced segment length
  - `equivalentSoundLevel_dBp`: Equivalent sound level in dB

### 2. Paralinguistic Features (25 dimensions)

Paralinguistic features capture the non-verbal aspects of speech that convey emotion, attitude, and cognitive state. These features are extracted by analyzing pause patterns, speech rate, and temporal characteristics of speech segments.

#### 2.1 Prosodic Features

##### Speech Rate (`speech_rate`)
- **Definition**: Number of words per second (assuming 1.5 syllables per word)
- **Formula**: $SR = \frac{N_{speech\_segments}}{T_{duration} \times 1.5}$
- **Physiological Significance**: Reflects motor control and cognitive processing speed
- **Cognitive Relevance**: Reduced speech rate may indicate cognitive slowing
- **Clinical Application**: Speech rate is decreased in dementia and Parkinson's disease

##### Articulation Rate (`articulation_rate`)
- **Definition**: Rate of speech excluding pauses
- **Formula**: $AR = \frac{N_{syllables}}{T_{speaking}}$
- **Physiological Significance**: Reflects motor speech control
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Articulation rate changes indicate motor speech disorders

##### Pause Duration Features
- **Definition**: Duration of silent intervals in speech, categorized by length
- **Formula**: $PD = \sum_{i=1}^{N} T_{pause_i}$
- **Physiological Significance**: Reflects planning and retrieval processes
- **Cognitive Relevance**: Increased pauses may indicate cognitive difficulty
- **Clinical Application**: Pause duration is increased in cognitive disorders
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
- **Cognitive Relevance**: Increased pause ratios may indicate cognitive load
- **Clinical Application**: Pause ratios are increased in cognitive disorders
- **Specific Features**:
  - `total_short_pause_duration_to_total_speech_duration`: Short pause ratio
  - `total_long_pause_duration_to_total_speech_duration`: Long pause ratio
  - `total_hesitation_duration_to_total_speech_duration`: Hesitation ratio
  - `silence_percentage`: Total silence percentage

#### 2.2 Speech Segment Features

##### Speech Segment Statistics
- **Definition**: Statistics of speech segments (segments between pauses)
- **Physiological Significance**: Reflects speech fluency and organization
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Speech segment changes indicate speech disorders
- **Specific Features**:
  - `speech_segment_count`: Number of speech segments
  - `avg_speech_duration`: Average speech segment duration
  - `max_speech_duration`: Maximum speech segment duration
  - `min_speech_duration`: Minimum speech segment duration

##### Phrase Segment Statistics
- **Definition**: Statistics of phrase segments (segments between hesitations)
- **Physiological Significance**: Reflects higher-level speech organization
- **Cognitive Relevance**: May indicate cognitive planning and organization
- **Clinical Application**: Phrase segment changes indicate cognitive disorders
- **Specific Features**:
  - `phrase_segment_count`: Number of phrase segments
  - `avg_phrase_duration`: Average phrase segment duration
  - `max_phrase_duration`: Maximum phrase segment duration
  - `min_phrase_duration`: Minimum phrase segment duration

##### Silence Statistics
- **Definition**: Overall silence and speaking time statistics
- **Physiological Significance**: Reflects overall speech efficiency
- **Cognitive Relevance**: May indicate cognitive processing efficiency
- **Clinical Application**: Silence statistics changes indicate speech disorders
- **Specific Features**:
  - `total_silence_duration`: Total silence duration
  - `total_speech_duration`: Total speaking duration
  - `silence_average_duration`: Average silence duration

#### 2.2 Rhythm and Timing Features

##### Rhythm Regularity (`rhythm_regularity`)
- **Definition**: Consistency of inter-syllable intervals
- **Formula**: $RR = 1 - \frac{\sigma_{ISI}}{\mu_{ISI}}$ where ISI is inter-syllable interval
- **Physiological Significance**: Reflects motor control and timing
- **Cognitive Relevance**: May indicate cognitive motor control
- **Clinical Application**: Rhythm regularity is decreased in Parkinson's disease

##### Syllable Duration Variability (`syllable_duration_var`)
- **Definition**: Variability in syllable durations
- **Formula**: $SDV = \frac{\sigma_{syllable\_duration}}{\mu_{syllable\_duration}}$
- **Physiological Significance**: Reflects motor control and timing precision
- **Cognitive Relevance**: May indicate cognitive motor control
- **Clinical Application**: Syllable duration variability is increased in cognitive disorders

##### Stress Pattern (`stress_pattern`)
- **Definition**: Pattern of stressed and unstressed syllables
- **Formula**: $SP = \frac{N_{stressed}}{N_{total}}$
- **Physiological Significance**: Reflects prosodic control and expression
- **Cognitive Relevance**: May indicate emotional expression and communication
- **Clinical Application**: Stress pattern changes indicate prosodic disorders

#### 2.3 Voice Quality Features

##### Jitter (`jitter`)
- **Definition**: Cycle-to-cycle variation in fundamental frequency
- **Formula**: $Jitter = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|F_0(i+1) - F_0(i)|}{F_0(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Cognitive Relevance**: May indicate neurological control
- **Clinical Application**: Jitter is increased in neurological disorders

##### Shimmer (`shimmer`)
- **Definition**: Cycle-to-cycle variation in amplitude
- **Formula**: $Shimmer = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|A(i+1) - A(i)|}{A(i)}$
- **Physiological Significance**: Reflects vocal cord stability
- **Cognitive Relevance**: May indicate neurological control
- **Clinical Application**: Shimmer is increased in neurological disorders

##### Harmonics-to-Noise Ratio (`HNR`)
- **Definition**: Ratio of harmonic to noise energy
- **Formula**: $HNR = 10\log_{10}\left(\frac{P_{harmonic}}{P_{noise}}\right)$
- **Physiological Significance**: Reflects voice quality and clarity
- **Cognitive Relevance**: May indicate vocal control and health
- **Clinical Application**: HNR is decreased in voice disorders

##### Glottal-to-Noise Excitation Ratio (`GNE`)
- **Definition**: Ratio of glottal to noise excitation
- **Formula**: $GNE = \frac{P_{glottal}}{P_{noise}}$
- **Physiological Significance**: Reflects voice source characteristics
- **Cognitive Relevance**: May indicate voice quality
- **Clinical Application**: GNE changes indicate voice quality variations

#### 2.4 Emotional and Affective Features

##### Emotional Valence (`emotional_valence`)
- **Definition**: Positive or negative emotional content
- **Formula**: $EV = \frac{1}{N}\sum_{i=1}^{N} valence_i$
- **Physiological Significance**: Reflects emotional state and expression
- **Cognitive Relevance**: May indicate mood and emotional regulation
- **Clinical Application**: Emotional valence changes indicate mood disorders

##### Emotional Arousal (`emotional_arousal`)
- **Definition**: Level of emotional activation
- **Formula**: $EA = \frac{1}{N}\sum_{i=1}^{N} arousal_i$
- **Physiological Significance**: Reflects emotional intensity
- **Cognitive Relevance**: May indicate emotional regulation
- **Clinical Application**: Emotional arousal changes indicate emotional disorders

##### Emotional Dominance (`emotional_dominance`)
- **Definition**: Level of emotional control
- **Formula**: $ED = \frac{1}{N}\sum_{i=1}^{N} dominance_i$
- **Physiological Significance**: Reflects emotional control and regulation
- **Cognitive Relevance**: May indicate emotional regulation
- **Clinical Application**: Emotional dominance changes indicate emotional disorders

### 3. Semantic Features (2 dimensions)

Semantic features capture the linguistic content and meaning of speech through automated speech recognition and natural language processing. These features are extracted using WeNet ASR framework and BERT-Chinese model to analyze semantic content and coverage of key elements in the Cookie Theft picture description task.

#### 3.1 Semantic Analysis Features

##### Match Score (`Match Score`)
- **Definition**: Semantic similarity between test speech and reference text using BERT embeddings
- **Formula**: $MatchScore = \frac{e_1 \cdot e_2}{||e_1|| \cdot ||e_2||}$ where $e_1, e_2 \in \mathbb{R}^{768}$
- **Physiological Significance**: Reflects semantic knowledge and language comprehension
- **Cognitive Relevance**: May indicate semantic memory and processing ability
- **Clinical Application**: Match score is decreased in cognitive disorders
- **Implementation**: Uses BERT-Chinese model to generate 768-dimensional sentence embeddings

##### Coverage Score (`Coverage Score`)
- **Definition**: Coverage of predefined key semantic elements in the Cookie Theft picture
- **Formula**: $CoverageScore = \frac{N_{covered\_elements}}{N_{total\_elements}}$
- **Physiological Significance**: Reflects attention to detail and semantic processing
- **Cognitive Relevance**: May indicate attention, memory, and semantic processing
- **Clinical Application**: Coverage score is decreased in cognitive disorders
- **Key Elements**: 19 semantic units including characters (男孩, 女孩, 女人), locations (家, 窗外), objects (饼干, 饼干盒, 凳子, 水池, 水龙头, 盘子, 毛巾, 桌子), and actions (站, 拿, 倒, 扶, 洗盘子, 水流出来)

<!-- 
##### Type-Token Ratio (`TTR`)
- **Definition**: Ratio of unique words to total words
- **Formula**: $TTR = \frac{N_{types}}{N_{tokens}}$
- **Physiological Significance**: Reflects lexical diversity
- **Cognitive Relevance**: May indicate cognitive reserve and language ability
- **Clinical Application**: TTR is decreased in cognitive disorders

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

2. **Paralinguistic Feature Extraction**:
   - Analyze pause patterns using energy threshold (0.1)
   - Categorize pauses: short (0.15-0.4s), long (0.4-1s), hesitation (>1s)
   - Compute 25-dimensional features including pause frequencies, durations, speech rate, segment statistics

3. **Semantic Feature Extraction**:
   - Use WeNet ASR for speech-to-text conversion
   - Apply BERT-Chinese model for semantic analysis
   - Compute Match Score (cosine similarity) and Coverage Score (key element coverage)

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

The comprehensive set of speech features described in this document provides a robust foundation for cognitive impairment assessment through speech analysis. These features capture multiple aspects of speech production and content including acoustic properties, paralinguistic characteristics, and semantic content.

When combined with appropriate signal processing techniques and clinical validation, these features can serve as valuable biomarkers for early detection, progression monitoring, and treatment evaluation in cognitive disorders. The integration of these features in a multi-modal framework, as implemented in the M3-CIA system, allows for comprehensive assessment of cognitive function using objective, quantifiable measures of speech production and content.

The extensive literature support and open-source code libraries, including OpenSMILE for acoustic analysis and BERT-Chinese for semantic analysis, provide researchers and clinicians with the tools necessary to implement and validate these features in their own studies, contributing to the advancement of cognitive health assessment through speech analysis.
