# sudo apt-get update
# sudo add-apt-repository universe
# sudo apt-get update
# sudo apt-get install ffmpeg

import subprocess
import os


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # Disable video
        "-acodec", "pcm_s16le",  # Audio codec
        "-ar", "16000",  # Audio sampling rate
        "-ac", "1",  # Mono audio
        "-y",  # Overwrite output file if it exists
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_to_wav(video_path):
    # Extract audio from video
    audio_path = video_path.replace('.mp4', '.wav')
    extract_audio(video_path, audio_path)

# Get all MP4 files in the current directory
mp4_files = [f for f in os.listdir() if f.endswith('.mp4')]

for mp4_file in mp4_files:
    print(f"Converting {mp4_file}...")
    wav_file = os.path.splitext(mp4_file)[0] + ".wav"
    convert_to_wav(mp4_file)
    print(f"Created {wav_file}")

print("All conversions completed.")