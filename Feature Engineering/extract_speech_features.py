import os
import glob
import argparse
import numpy as np
import pandas as pd
import re
import librosa
import opensmile
import jieba
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Preprocessing Functions
# ==========================================

def preprocess_audio(file_path, target_sr=16000):
    """
    Load and preprocess audio suitable for speech feature extraction.
    Resamples to target_sr and normalizes the waveform.
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    # Normalize amplitude
    y = librosa.util.normalize(y)
    return y, sr

# ==========================================
# 2. Acoustic & Voice Quality Features (Spectral, Prosodic, Voice Quality)
# ==========================================

def extract_opensmile_features(file_path):
    """
    Extract Spectral, Prosodic, and Voice Quality features using OpenSMILE (eGeMAPS).
    This covers MFCCs, Formants, F0, Loudness, Jitter, Shimmer, HNR, Flux, etc.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(file_path)
    # Convert single-row dataframe to dictionary
    feature_dict = df.iloc[0].to_dict()
    
    # Optional: Rename or select specific keys to match the MD precisely, 
    # but eGeMAPS natively includes all standard markers (F0, Jitter, Shimmer, HNR, MFCC).
    return feature_dict

def extract_librosa_spectral(y, sr):
    """
    Supplementary spectral features not fully covered by basic functionals.
    """
    features = {}
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(cent)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(rolloff)

    # Spectral Bandwidth and Contrast
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(contrast)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['ZCR_mean'] = np.mean(zcr)

    try:
        f0 = librosa.yin(y, fmin=50, fmax=min(500, sr / 2 - 1), sr=sr)
        f0 = f0[np.isfinite(f0)]
        if len(f0) > 1:
            features['F0_range'] = np.max(f0) - np.min(f0)
            features['F0_slope'] = np.polyfit(np.arange(len(f0)), f0, 1)[0]
    except Exception:
        pass
    
    return features

# ==========================================
# 3. Fluency & Temporal Features
# ==========================================

def extract_fluency_features(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    Extract Fluency and Temporal features using voice activity detection (VAD).
    """
    features = {}
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Split audio into non-silent intervals
    non_mute_intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    if len(non_mute_intervals) == 0:
        return features

    # Calculate durations
    speech_durations = [(end - start) / sr for start, end in non_mute_intervals]
    total_speech_duration = sum(speech_durations)
    total_silence_duration = total_duration - total_speech_duration
    
    # Phonation Ratio
    features['phonation_ratio'] = total_speech_duration / total_duration if total_duration > 0 else 0
    features['total_speech_duration'] = total_speech_duration
    
    # Pause (Silence) Analysis
    # Pauses are the gaps between non-mute intervals
    pauses = []
    for i in range(1, len(non_mute_intervals)):
        pause_start = non_mute_intervals[i-1][1]
        pause_end = non_mute_intervals[i][0]
        pause_dur = (pause_end - pause_start) / sr
        if pause_dur > 0.15: # Ignore very tiny micro-pauses
            pauses.append(pause_dur)
            
    features['pause_count'] = len(pauses)
    features['total_pause_duration'] = sum(pauses)
    features['average_pause_duration'] = np.mean(pauses) if pauses else 0
    features['pause_ratio'] = features['total_pause_duration'] / total_duration if total_duration > 0 else 0
    
    # Short vs Long Pauses
    short_pauses = [p for p in pauses if 0.15 <= p < 0.4]
    long_pauses = [p for p in pauses if 0.4 <= p < 1.0]
    hesitation_pauses = [p for p in pauses if p >= 1.0]
    features['short_pause_frequency'] = len(short_pauses) / total_duration if total_duration > 0 else 0
    features['long_pause_frequency'] = len(long_pauses) / total_duration if total_duration > 0 else 0
    features['hesitation_frequency'] = len(hesitation_pauses) / total_duration if total_duration > 0 else 0
    features['total_short_pause_duration'] = sum(short_pauses)
    features['total_long_pause_duration'] = sum(long_pauses)
    features['total_hesitation_duration'] = sum(hesitation_pauses)
    features['average_short_pause_duration'] = np.mean(short_pauses) if short_pauses else 0
    features['average_long_pause_duration'] = np.mean(long_pauses) if long_pauses else 0
    features['average_hesitation_duration'] = np.mean(hesitation_pauses) if hesitation_pauses else 0
    features['total_short_pause_duration_to_total_speech_duration'] = (
        features['total_short_pause_duration'] / total_speech_duration if total_speech_duration > 0 else 0
    )
    features['total_long_pause_duration_to_total_speech_duration'] = (
        features['total_long_pause_duration'] / total_speech_duration if total_speech_duration > 0 else 0
    )
    features['total_hesitation_duration_to_total_speech_duration'] = (
        features['total_hesitation_duration'] / total_speech_duration if total_speech_duration > 0 else 0
    )
    features['silence_percentage'] = total_silence_duration / total_duration if total_duration > 0 else 0

    # Speech Segments
    features['speech_segment_count'] = len(speech_durations)
    features['average_speech_segment_duration'] = np.mean(speech_durations) if speech_durations else 0
    features['max_speech_segment_duration'] = np.max(speech_durations) if speech_durations else 0

    return features

# ==========================================
# 4. Semantic Features
# ==========================================

# Cookie Theft Picture standardized keywords

TARGET_KEYWORDS = [
    "\u7537\u5b69", "\u5973\u5b69", "\u5973\u4eba", "\u5988\u5988", "\u53a8\u623f", "\u7a97\u5916", "\u997c\u5e72", "\u997c\u5e72\u76d2",
    "\u51f3\u5b50", "\u6c34\u6c60", "\u6c34\u9f99\u5934", "\u76d8\u5b50", "\u6bdb\u5dfe", "\u684c\u5b50", "\u7ad9", "\u62ff",
    "\u5012", "\u6454", "\u6d17\u76d8\u5b50", "\u6c34\u6d41\u51fa\u6765", "\u6454\u5012"
]

def extract_semantic_features(text, tokenizer=None, model=None, reference_embedding=None):
    """
    Extract Semantic features from textual transcript.
    Includes TTR, Coverage Score, and BERT Match Score.
    """
    features = {}
    if not text or not isinstance(text, str):
        return features

    # Tokenization using jieba
    words = list(jieba.cut(text))
    words = [w for w in words if w.strip()] # Remove whitespaces
    
    if not words:
        return features

    # TTR (Type-Token Ratio)
    unique_words = set(words)
    features['TTR'] = len(unique_words) / len(words)
    
    # Word count and derived fluency
    features['total_words'] = len(words)
    sentence_units = [s for s in re.split(r'[\u3002\uff01\uff1f!?;\uff1b]+', text) if s.strip()]
    if sentence_units:
        sentence_lengths = [len([w for w in jieba.cut(s) if w.strip()]) for s in sentence_units]
        features['syntax_depth'] = np.mean(sentence_lengths)
    else:
        features['syntax_depth'] = len(words)

    # Coverage Score (Cookie Theft)
    covered_items = [kw for kw in TARGET_KEYWORDS if kw in text]
    features['Coverage_Score'] = len(covered_items) / len(TARGET_KEYWORDS)
    features['Coverage Score'] = features['Coverage_Score']
    features['Covered_Elements_Count'] = len(covered_items)

    # Hesitation Ratio (filled pauses)
    hesitations = ["\u5443", "\u55ef", "\u554a", "\u90a3\u4e2a", "\u5c31\u662f"]
    hesitation_count = sum(1 for w in words if w in hesitations)
    features['hesitation_ratio'] = hesitation_count / len(words)

    # BERT Embeddings Semantic Similarity (Match Score)
    if tokenizer and model and reference_embedding is not None:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Pooler output or mean of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        sim = cosine_similarity(embeddings, reference_embedding)[0][0]
        features['BERT_Match_Score'] = sim
        features['Match Score'] = sim

    return features

# ==========================================
# 5. Main Execution Pipeline
# ==========================================

def get_reference_bert_embedding(tokenizer, model):
    """Create a reference embedding mimicking a perfect Cookie Theft description."""
    reference_text = (
        "\u8fd9\u662f\u4e00\u4e2a\u53a8\u623f\u7684\u573a\u666f\u3002\u4e00\u4e2a\u5973\u4eba\u6b63\u7ad9\u5728\u6c34\u6c60\u524d\u6d17\u76d8\u5b50\uff0c\u6c34\u9f99\u5934\u6ca1\u6709\u5173\uff0c"
        "\u6c34\u6d41\u51fa\u6765\u6ea2\u5230\u4e86\u5730\u4e0a\uff0c\u5979\u5374\u6ca1\u6709\u53d1\u73b0\u3002\u5979\u65c1\u8fb9\u6709\u4e24\u4e2a\u5b69\u5b50\uff0c\u4e00\u4e2a\u7537\u5b69\u7ad9\u5728"
        "\u6447\u6643\u7684\u51f3\u5b50\u4e0a\u6b63\u4ece\u6a71\u67dc\u91cc\u62ff\u997c\u5e72\uff0c\u4ed6\u9012\u7ed9\u4e0b\u9762\u7684\u5973\u5b69\u3002\u7a97\u5916\u53ef\u4ee5\u770b\u5230\u9662\u5b50\u3002"
    )
    inputs = tokenizer(reference_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

def process_single_speech(audio_path, transcript=None, bert_toolkit=None):
    """
    Process a single audio file and its transcript to extract all features.
    """
    feats = {}
    
    # 1. Acoustic / Prosodic / Voice Quality via OpenSMILE
    # Very comprehensive out of the box
    try:
        os_feats = extract_opensmile_features(audio_path)
        feats.update(os_feats)
    except Exception as e:
        print(f"Warning: OpenSMILE extraction failed for {audio_path}. {e}")

    # 2. Load audio for extra Fluency and Librosa spectral
    try:
        y, sr = preprocess_audio(audio_path)
        feats.update(extract_librosa_spectral(y, sr))
        
        fluency_feats = extract_fluency_features(y, sr)
        feats.update(fluency_feats)
        
    except Exception as e:
        print(f"Warning: Librosa/Fluency extraction failed for {audio_path}. {e}")

    # 3. Semantic Features
    if transcript:
        try:
            tokenizer, model, ref_emb = bert_toolkit if bert_toolkit else (None, None, None)
            sem_feats = extract_semantic_features(transcript, tokenizer, model, ref_emb)
            feats.update(sem_feats)
        except Exception as e:
            print(f"Warning: Semantic extraction failed for {transcript}. {e}")

    if transcript and 'total_words' in feats:
        speaking_time = feats.get('total_speech_duration', 0)
        total_duration = librosa.get_duration(y=y, sr=sr) if 'y' in locals() else 0
        feats['speech_rate'] = feats['total_words'] / speaking_time if speaking_time > 0 else 0
        feats['articulation_rate'] = feats['speech_rate']
        feats['speech_rate_total_duration'] = feats['total_words'] / total_duration if total_duration > 0 else 0

    return feats

def main():
    parser = argparse.ArgumentParser(description="Standardized Speech Feature Extraction for Cognitive Impairment Assessment")
    parser.add_argument('--audio_dir', type=str, required=True, help="Directory containing input (.wav/.mp3) files")
    parser.add_argument('--text_dir', type=str, default=None, help="Directory containing transcripts (.txt) matching audio filenames")
    parser.add_argument('--output_file', type=str, default='speech_features_output.csv', help="Output CSV file path")
    parser.add_argument('--use_bert', action='store_true', help="Enable BERT for Semantic Match Score")
    
    args = parser.parse_args()
    
    audio_files = glob.glob(os.path.join(args.audio_dir, '*.*')) # wav, mp3, etc.
    audio_files = [f for f in audio_files if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return

    # Load BERT if requested (Requires transformers)
    bert_toolkit = None
    if args.use_bert:
        try:
            print("Loading BERT model...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            model = BertModel.from_pretrained('bert-base-chinese')
            ref_emb = get_reference_bert_embedding(tokenizer, model)
            bert_toolkit = (tokenizer, model, ref_emb)
        except Exception as e:
            print(f"Failed to load BERT: {e}. Match Score will be skipped.")

    all_features = []
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        base_name = os.path.splitext(filename)[0]
        print(f"Processing: {filename}...")
        
        transcript = ""
        if args.text_dir:
            text_path = os.path.join(args.text_dir, f"{base_name}.txt")
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()

        feats = process_single_speech(audio_path, transcript, bert_toolkit)
        feats['File_Name'] = filename
        
        # Move File_Name to front
        feats = {'File_Name': feats.pop('File_Name'), **feats}
        all_features.append(feats)

    if all_features:
        df_out = pd.DataFrame(all_features)
        df_out.to_csv(args.output_file, index=False)
        print(f"Successfully processed {len(all_features)} files. Results saved to {args.output_file}")
    else:
        print("No valid features extracted from any file.")

if __name__ == "__main__":
    main()
