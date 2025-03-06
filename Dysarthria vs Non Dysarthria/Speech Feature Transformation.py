import pandas as pd
import Speech_Feature_Extraction

folder_path = "D:\MS Data Science\Final MSc Project\Project Code\Dysarthria and Non Dysarthria\Female_Non_Dysarthria\FC03\Session3\Wav" ##Folder path of actual raw audio files
audio_files = Speech_Feature_Extraction.list_audio_files(folder_path)

features = Speech_Feature_Extraction.process_audio_files(audio_files)

df_features = pd.DataFrame(features)
df_features.columns = [f'feature_{i}' for i in range(df_features.shape[1])]

df_features['label'] = 0

df_features.to_csv('features_female_non_3.csv', index=False)

print(df_features.head())