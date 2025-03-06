import os
import torchaudio
import librosa
import numpy as np
import pandas as pd


def load_audio(file_path, sample_rate=16000):
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        waveform = waveform.numpy().flatten().astype(np.float32)
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_features(audio, sample_rate, max_length=400):
    try:
        if len(audio) < 2048:  # Skip files that are too short
            print(f"Audio file too short: {len(audio)} samples")
            return None

        n_fft = min(2048, len(audio))

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft)

        features = np.concatenate((mfcc, chroma), axis=0)

        if features.shape[1] > max_length:
            features = features[:, :max_length]
        else:
            pad_width = max_length - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return features
    except Exception as e:
        print(f"Error extracting features from audio: {e}")
        return None


def list_audio_files(directory, extension=".wav"):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio_files(file_paths, sample_rate=16000, max_length=400):
    feature_list = []
    for file_path in file_paths:
        audio, sr = load_audio(file_path, sample_rate)
        if audio is not None and sr is not None:
            features = extract_features(audio, sr, max_length)
            if features is not None:
                feature_list.append(features.flatten())
    return np.array(feature_list)



