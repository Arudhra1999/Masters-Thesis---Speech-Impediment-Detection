import os
import pandas as pd
import librosa
import numpy as np


data = pd.read_csv('D:\\Time Series EDA\\SEP-28k_labels - 1.csv')

# Define base directory where the audio files are stored
base_audio_dir = 'Clipped_Audio'
# Create lists to store MFCC features and labels
mfcc_features = []
labels = []

# Iterate over the rows in the CSV file
for index, row in data.iterrows():
    folder_name = row['Show']
    audio_id = row['EpId']
    clip_id = row['ClipId']
    label = row['Original Label']

    # Construct the file path
    audio_path = os.path.join(base_audio_dir, folder_name, f"{audio_id}", f"{folder_name}_{audio_id}_{clip_id}.wav")

    # Load the audio clip
    if os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=16000)

        # Compute MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

        # Average MFCCs over time
        mfcc_mean = np.mean(mfcc, axis=1)

        # Store the MFCC features and label
        mfcc_features.append(mfcc_mean.tolist())
        labels.append(label)
        print(f'MFCC extracted for {audio_path}')
    else:
        print(f"File not found: {audio_path}")

# Create a DataFrame for MFCC features and labels
df_mfcc = pd.DataFrame(mfcc_features)
df_mfcc['Label'] = labels
df_mfcc.to_csv('mfcc_features_with_labels_4.csv', index=False)
