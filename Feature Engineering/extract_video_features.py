import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy

# ==========================================
# 1. OpenFace Output Parser & Preprocessor
# ==========================================

def load_openface_csv(csv_path, confidence_threshold=0.7):
    """
    Load OpenFace feature CSV and filter out low-confidence frames.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    # Strip whitespace from column names if any
    df.columns = [col.strip() for col in df.columns]
    
    # Filter valid frames
    if 'confidence' in df.columns and 'success' in df.columns:
        valid_df = df[(df['success'] == 1) & (df['confidence'] >= confidence_threshold)].copy()
    else:
        valid_df = df.copy()
        
    return valid_df

# ==========================================
# 2. Action Units (AUs) Feature Extraction
# ==========================================

def extract_au_features(df):
    """
    Extract facial muscle action features (AUs) based on MD specifications.
    Calculates the mean intensity (_r) for Upper, Mid, Lower face, and Eyes.
    """
    features = {}
    if df.empty:
        return features

    # Define requested AUs from the MD file
    target_aus = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
        'AU18_r', 'AU20_r', 'AU23_r', 'AU24_r', 'AU25_r', 'AU26_r', 
        'AU27_r', 'AU43_r', 'AU45_r'
    ]

    for au in target_aus:
        if au in df.columns:
            features[f'{au}_mean'] = df[au].mean()
            features[f'{au}_std'] = df[au].std()
            features[f'{au}_max'] = df[au].max()
        else:
            # Handle missing columns gracefully
            features[f'{au}_mean'] = np.nan
            features[f'{au}_std'] = np.nan
            features[f'{au}_max'] = np.nan

    present_au_cols = [au for au in target_aus if au in df.columns]
    if present_au_cols:
        au_values = df[present_au_cols].clip(lower=0).to_numpy(dtype=float)
        active_mask = au_values > 0.5
        frame_active_counts = active_mask.sum(axis=1)
        active_intensity_sum = np.where(active_mask, au_values, 0).sum(axis=1)

        features['AUP'] = active_mask.mean()
        for au, au_active in zip(present_au_cols, active_mask.T):
            features[f'{au.replace("_r", "_c")}_AUP'] = au_active.mean()

        active_frame_mask = frame_active_counts > 0
        features['AUI'] = (
            np.mean(active_intensity_sum[active_frame_mask] / frame_active_counts[active_frame_mask])
            if np.any(active_frame_mask) else 0.0
        )

        au_distribution = au_values.sum(axis=0)
        if np.sum(au_distribution) > 0:
            features['expressive_entropy'] = entropy(au_distribution / np.sum(au_distribution), base=2)
        else:
            features['expressive_entropy'] = 0.0

    return features

# ==========================================
# 3. Head Motor Dynamics
# ==========================================

def extract_head_dynamics(df):
    """
    Extract head pose, velocity, jerk, and nodding frequency from OpenFace data.
    """
    features = {}
    if df.empty or len(df) < 5:
        return features

    # Time array for derivatives
    t = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) / 30.0
    dt = np.diff(t)
    dt[~np.isfinite(dt) | (dt <= 0)] = 0.033  # fallback to ~30fps

    # 1. Head Pose (Pitch, Yaw, Roll)
    pose_cols = ['pose_Rx', 'pose_Ry', 'pose_Rz']
    for col in pose_cols:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()

    # Need Tx, Ty, Tz for translation kinematics
    trans_cols = ['pose_Tx', 'pose_Ty', 'pose_Tz']
    if all(c in df.columns for c in trans_cols) and all(c in df.columns for c in pose_cols):
        tx, ty, tz = df['pose_Tx'].values, df['pose_Ty'].values, df['pose_Tz'].values
        rx, ry, rz = df['pose_Rx'].values, df['pose_Ry'].values, df['pose_Rz'].values

        # 2. Head Velocity (1st derivative)
        vx = np.diff(tx) / dt
        vy = np.diff(ty) / dt
        vz = np.diff(tz) / dt
        vrx = np.diff(rx) / dt
        vry = np.diff(ry) / dt
        vrz = np.diff(rz) / dt

        trans_vel = np.sqrt(vx**2 + vy**2 + vz**2)
        rot_vel = np.sqrt(vrx**2 + vry**2 + vrz**2)

        features['head_translation_velocity_mean'] = trans_vel.mean()
        features['head_rotation_velocity_mean'] = rot_vel.mean()
        features['head_velocity'] = np.mean(trans_vel + rot_vel)

        # 3. Head Jerk (3rd derivative of position)
        ax = np.diff(vx) / dt[:-1]
        ay = np.diff(vy) / dt[:-1]
        az = np.diff(vz) / dt[:-1]

        jx = np.diff(ax) / dt[:-2]
        jy = np.diff(ay) / dt[:-2]
        jz = np.diff(az) / dt[:-2]

        jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)
        features['head_jerk_mean'] = jerk_mag.mean() if len(jerk_mag) > 0 else np.nan
        features['head_jerk'] = features['head_jerk_mean']

        # 4. Nodding Frequency (Oscillations in Pitch / pose_Rx)
        # Smooth pitch to remove high-frequency noise
        smoothed_pitch = savgol_filter(rx, window_length=min(11, len(rx)-1 if len(rx)%2==0 else len(rx)), polyorder=2)
        peaks, _ = find_peaks(smoothed_pitch, prominence=0.05) # 0.05 radians ~ 2.8 degrees
        total_time = t[-1] - t[0] if len(t) > 1 else 1

        features['nodding_frequency'] = len(peaks) / total_time if total_time > 0 else 0

    return features

# ==========================================
# 4. Eye Movement Patterns
# ==========================================

def extract_eye_movements(df):
    """
    Extract gaze angles, blink rate, and fixation duration.
    """
    features = {}
    if df.empty:
        return features
        
    t = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) / 30.0
    total_time_mins = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 1.0

    # 1. Gaze Angle
    gaze_cols = ['gaze_angle_x', 'gaze_angle_y']
    for col in gaze_cols:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()

    # 2. Blink Rate (using AU45_c or AU45_r)
    if 'AU45_c' in df.columns:
        # Count rising edges of the binary blink classifier
        blink_signal = df['AU45_c'].values
        blinks = np.sum(np.diff(blink_signal) == 1)
        features['blink_rate_per_min'] = blinks / total_time_mins if total_time_mins > 0 else 0
        features['blink_rate'] = features['blink_rate_per_min']
    elif 'AU45_r' in df.columns:
        # Fallback to finding peaks in the continuous blink signal
        peaks, _ = find_peaks(df['AU45_r'].values, height=2.0, distance=5)
        features['blink_rate_per_min'] = len(peaks) / total_time_mins if total_time_mins > 0 else 0
        features['blink_rate'] = features['blink_rate_per_min']

    # 3. Fixation Duration
    # A fixation is defined as periods where angular velocity of gaze is VERY small.
    if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns and len(t) > 1:
        dt = np.diff(t)
        dt[~np.isfinite(dt) | (dt <= 0)] = 0.033
        
        gx = df['gaze_angle_x'].values
        gy = df['gaze_angle_y'].values
        
        gaze_vel_x = np.diff(gx) / dt
        gaze_vel_y = np.diff(gy) / dt
        gaze_angular_vel = np.sqrt(gaze_vel_x**2 + gaze_vel_y**2)
        
        # Threshold for fixation (e.g., velocity < 0.5 rad/s approx 28 deg/s)
        fixation_mask = gaze_angular_vel < 0.5 
        
        # Calculate duration of consecutive True values in fixation_mask
        fixation_durations = []
        current_dur = 0
        for i, is_fix in enumerate(fixation_mask):
            if is_fix:
                current_dur += dt[i]
            else:
                if current_dur > 0.1: # Minimum fixation time ~ 100ms
                    fixation_durations.append(current_dur)
                current_dur = 0
                
        if current_dur > 0.1:
            fixation_durations.append(current_dur)
            
        features['fixation_duration_mean'] = np.mean(fixation_durations) if fixation_durations else 0.0
        features['fixation_duration_std'] = np.std(fixation_durations) if fixation_durations else 0.0
        features['fixation_duration'] = features['fixation_duration_mean']

    return features

# ==========================================
# 5. Facial Kinematics & Geometry
# ==========================================

def extract_facial_kinematics(df):
    """
    Extract landmark velocity and a simple landmark-based asymmetry index.
    OpenFace stores 2D landmarks as x_0..x_67 and y_0..y_67.
    """
    features = {}
    if df.empty or len(df) < 2:
        return features

    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    if not all(c in df.columns for c in x_cols + y_cols):
        return features

    t = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) / 30.0
    dt = np.diff(t)
    dt[~np.isfinite(dt) | (dt <= 0)] = 0.033

    x = df[x_cols].to_numpy(dtype=float)
    y = df[y_cols].to_numpy(dtype=float)

    dx = np.diff(x, axis=0) / dt[:, None]
    dy = np.diff(y, axis=0) / dt[:, None]
    landmark_speed = np.sqrt(dx**2 + dy**2)
    features['landmark_velocity'] = np.nanmean(landmark_speed)

    left_right_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (31, 35), (32, 34), (36, 45), (37, 44), (38, 43), (39, 42),
        (40, 47), (41, 46), (48, 54), (49, 53), (50, 52), (55, 59), (56, 58)
    ]
    center_x = x[:, 30]
    face_width = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    face_width[face_width <= 0] = np.nan
    pair_asymmetry = []
    for left, right in left_right_pairs:
        mirrored_distance = np.abs((x[:, left] - center_x) + (x[:, right] - center_x))
        pair_asymmetry.append(mirrored_distance / face_width)
    features['asymmetry_index'] = np.nanmean(pair_asymmetry)

    return features

# ==========================================
# 6. Main Processing Pipeline
# ==========================================

def process_video_csv(csv_path):
    """
    Process a single OpenFace CSV to extract all features.
    """
    df = load_openface_csv(csv_path)
    
    feats = {}
    feats.update(extract_au_features(df))
    feats.update(extract_head_dynamics(df))
    feats.update(extract_eye_movements(df))
    feats.update(extract_facial_kinematics(df))
    
    return feats

def main():
    parser = argparse.ArgumentParser(description="Standardized Video Feature Extraction for Cognitive Impairment")
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Directory containing input OpenFace feature files (.csv)")
    parser.add_argument('--output_file', type=str, default='video_features_output.csv', 
                        help="Path for the output extracted features CSV")
    
    # Optional arguments to run OpenFace directly (if you have videos instead of CSVs)
    parser.add_argument('--video_dir', type=str, default=None, 
                        help="Optional: Directory of raw videos. If provided, OpenFace will run first.")
    parser.add_argument('--openface_path', type=str, default="FeatureExtraction", 
                        help="Path to the OpenFace 'FeatureExtraction' executable")

    args = parser.parse_args()

    # 1. (Optional) Run OpenFace on raw videos if requested
    if args.video_dir:
        print(f"Running OpenFace on videos in {args.video_dir}...")
        video_files = glob.glob(os.path.join(args.video_dir, '*.mp4')) + glob.glob(os.path.join(args.video_dir, '*.avi'))
        for vid in video_files:
            cmd = f'"{args.openface_path}" -f "{vid}" -out_dir "{args.input_dir}"'
            print(f"Executing: {cmd}")
            os.system(cmd)
            
    # 2. Process CSV files
    csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}. Please ensure OpenFace has processed the videos.")
        return

    all_features = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"Extracting features from: {filename}...")
        try:
            feats = process_video_csv(csv_path)
            feats['File_Name'] = filename
            
            # Reorder to put File_Name first
            feats = {'File_Name': feats.pop('File_Name'), **feats}
            all_features.append(feats)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 3. Save Output
    if all_features:
        df_out = pd.DataFrame(all_features)
        df_out.to_csv(args.output_file, index=False)
        print(f"\nSuccessfully processed {len(all_features)} files.")
        print(f"Comprehensive video features saved to: {args.output_file}")
    else:
        print("No valid features extracted from any file.")

if __name__ == "__main__":
    main()
