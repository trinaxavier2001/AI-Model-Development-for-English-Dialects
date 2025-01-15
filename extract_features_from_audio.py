import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing

# Define padding and truncation function
def pad_or_truncate_audio(y, sr, target_length=10):
    max_length = target_length * sr  # Convert target length to samples
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), mode='constant')  # Pad with zeros
    else:
        y = y[:max_length]  # Truncate to target length
    return y

# Function to extract features
def extract_features(y, sr):
    features = {}
    # 1. MFCCs
    features['mfccs'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).flatten()
    # 2. Mel Spectrogram
    features['mel_spectrogram'] = librosa.feature.melspectrogram(y=y, sr=sr).flatten()
    # 3. Chroma Features
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
    # 4. Spectrogram
    features['spectrogram'] = np.abs(librosa.stft(y)).flatten()
    return features

# Process a single file (wrapped for parallelization)
def process_file(file_path, label, target_length=10):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        # Pad or truncate the audio
        y = pad_or_truncate_audio(y, sr, target_length=target_length)
        # Extract features
        features = extract_features(y, sr)
        # Add label
        features['label'] = label
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Process audio files in directories
def process_audio_files(base_dir, output_dir, target_length=10, n_jobs=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through folders in the base directory
    folder_list = os.listdir(base_dir)
    for folder in tqdm(folder_list, desc="Processing Folders"):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a folder
            label = folder  # The folder name serves as the label (e.g., "irish_male")
            print(f"\nProcessing folder: {label}")
            
            # List all .wav files in the folder
            file_list = [
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith('.wav')
            ]
            
            # Parallel processing of files
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_file)(file_path, label, target_length)
                for file_path in tqdm(file_list, desc=f"Processing Files in {label}", leave=False)
            )
            
            # Remove None entries (failed files)
            results = [res for res in results if res is not None]
            
            # Save the features for this folder into a CSV file
            if results:
                output_file = os.path.join(output_dir, f"{label}_features.csv")
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"Saved features for {label} to {output_file}")

# Set base directory and output directory
base_dir = "audio_files"  # Replace with the path to your audio files
output_dir = "output_features"  # Directory to save the features

# Process the audio files with parallelization
process_audio_files(base_dir, output_dir, target_length=10, n_jobs=4)  # Adjust n_jobs based on your CPU cores