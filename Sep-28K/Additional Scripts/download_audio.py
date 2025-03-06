import os
import pandas as pd
import requests
from pydub import AudioSegment

# Load the CSV file
csv_file_path = 'D:\Time Series EDA\SEP-28k_episodes.csv'  # Path to your CSV file
# Define appropriate column names
column_names = ["Podcast_Name", "Episode_Name", "URL", "Folder_Name", "File_Number"]
data = pd.read_csv(csv_file_path, header=None, names=column_names)

# Display the first few rows of the DataFrame to ensure correctness
print(data.head())

# Ensure the base directory exists
base_dir = 'audio_files'  # Update this path to your desired base directory
os.makedirs(base_dir, exist_ok=True)

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def convert_to_wav(audio_path, wav_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")

# Iterate over each row and process the audio files
for index, row in data.iterrows():
    folder_name = row['Folder_Name'].strip()
    file_number = row['File_Number']
    url = row['URL'].strip()

    # Create the directory if it doesn't exist
    dir_path = os.path.join(base_dir, folder_name)
    os.makedirs(dir_path, exist_ok=True)

    # Set file paths
    audio_path_orig = os.path.join(dir_path, f"{file_number}.mp3")
    wav_path = os.path.join(dir_path, f"{file_number}.wav")

    # Download the audio file
    if not os.path.exists(audio_path_orig):
        print(f"Downloading {url} to {audio_path_orig}")
        try:
            download_file(url, audio_path_orig)
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
            continue

    # Convert to 16kHz mono WAV file
    print(f"Converting {audio_path_orig} to {wav_path}")
    try:
        convert_to_wav(audio_path_orig, wav_path)
    except Exception as e:
        print(f"Failed to convert {audio_path_orig} to WAV: {e}")
        continue

    # Remove the original file
    if os.path.exists(audio_path_orig):
        os.remove(audio_path_orig)
        print(f"Removed original file {audio_path_orig}")

print("All files processed.")
