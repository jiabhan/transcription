# pip install ctranslate2 # cuda 12
# pip install faster-whisper

import subprocess
import os
import datetime

from faster_whisper import WhisperModel


cwd = os.getcwd()

# Load the Whisper model
model = WhisperModel("large-v3", device="cuda", download_root=cwd)

def format_time(seconds):
    time = str(datetime.timedelta(seconds=seconds))
    if '.' in time:
        time = time[:-3]  # Remove microseconds
    else:
        time += '.000'  # Add milliseconds if not present
    return time.replace('.', ',').zfill(12)  # Ensure 12 characters with leading zeros

def convert_to_srt(video_path, output_path):

    # Extract audio from video
    audio_path = video_path.replace('.mp4', '.wav')

    # Transcribe the audio
    segments, _ = model.transcribe(audio_path)

    # Convert segments to SRT format
    srt_content = ""
    for i, segment in enumerate(segments, start=1):
        start = format_time(segment.start)
        end = format_time(segment.end)
        text = segment.text.strip()
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)


# Get all WAV files in the current directory
wav_files = [f for f in os.listdir() if f.endswith('.wav')]

for wav_file in wav_files:
    print(f"Converting {wav_file}...")
    srt_file = os.path.splitext(wav_file)[0] + ".srt"
    convert_to_srt(wav_file, srt_file)
    print(f"Created {srt_file}")

print("All conversions completed.")