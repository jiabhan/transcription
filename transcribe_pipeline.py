import subprocess
import os
import datetime
from faster_whisper import WhisperModel
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_pipeline.log'),
        logging.StreamHandler()
    ]
)


def verify_file_exists(file_path):
    """Verify file exists and is accessible."""
    path = Path(file_path)
    try:
        return path.exists() and os.access(path, os.R_OK)
    except Exception as e:
        logging.error(f"Error checking file {file_path}: {str(e)}")
        return False


def extract_audio(video_path, audio_path, max_retries=3):
    """Extract audio with retry logic and verification."""
    for attempt in range(max_retries):
        try:
            if not verify_file_exists(video_path):
                raise FileNotFoundError(f"Video file not found or not accessible: {video_path}")

            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                audio_path
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise subprocess.SubprocessError(f"FFmpeg error: {result.stderr}")

            if not verify_file_exists(audio_path):
                raise FileNotFoundError("Audio file was not created")

            return True

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} failed for {video_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False


def format_time(seconds):
    """Format time for SRT files."""
    try:
        time = str(datetime.timedelta(seconds=float(seconds)))
        if '.' in time:
            time = time[:-3]
        else:
            time += '.000'
        return time.replace('.', ',').zfill(12)
    except Exception as e:
        logging.error(f"Error formatting time {seconds}: {str(e)}")
        return "00:00:00,000"


def extract_text_from_srt(input_file, output_file):
    """Extract text from SRT with error handling."""
    try:
        if not verify_file_exists(input_file):
            raise FileNotFoundError(f"SRT file not found: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        blocks = content.strip().split('\n\n')
        text_only = []

        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:  # Verify block structure
                text_lines = lines[2:]
                text_only.append(' '.join(text_lines))

        extracted_text = ' '.join(text_only)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(extracted_text)

        logging.info(f"Successfully extracted text: '{output_file}'")

    except Exception as e:
        logging.error(f"Error processing '{input_file}': {str(e)}")
        return False

    return True


def find_all_videos(root_folder):
    """Find all video files with error handling."""
    video_files = []
    try:
        for root, _, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    if verify_file_exists(video_path):
                        video_files.append(video_path)
                    else:
                        logging.warning(f"Found video file but cannot access: {video_path}")
    except Exception as e:
        logging.error(f"Error walking directory {root_folder}: {str(e)}")

    return video_files


def transcribe_audio(model, wav_path, srt_path, max_retries=3):
    """Transcribe audio with retry logic."""
    for attempt in range(max_retries):
        try:
            if not verify_file_exists(wav_path):
                raise FileNotFoundError(f"WAV file not found: {wav_path}")

            segments, _ = model.transcribe(wav_path)
            srt_content = ""
            for i, segment in enumerate(segments, start=1):
                start = format_time(segment.start)
                end = format_time(segment.end)
                text = segment.text.strip()
                srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

            os.makedirs(os.path.dirname(srt_path), exist_ok=True)
            with open(srt_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(srt_content)

            return True

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} failed for {wav_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False


def main():
    try:
        # Get input folder from user
        root_folder = input("Enter the root folder path: ").strip()

        if not os.path.exists(root_folder):
            logging.error("Error: Specified folder does not exist!")
            return

        # Find all video files
        logging.info("Finding all video files...")
        video_files = find_all_videos(root_folder)
        logging.info(f"Found {len(video_files)} video files")

        if not video_files:
            logging.error("No video files found!")
            return

        # Track successful and failed operations
        results = {
            'audio_extraction': {'success': 0, 'failed': 0},
            'transcription': {'success': 0, 'failed': 0},
            'text_extraction': {'success': 0, 'failed': 0}
        }

        # Phase 1: Extract all audio
        logging.info("Phase 1: Extracting audio from all videos...")
        wav_files = []
        for video_path in video_files:
            wav_path = os.path.splitext(video_path)[0] + '.wav'
            logging.info(f"Extracting audio: {video_path}")

            if extract_audio(video_path, wav_path):
                wav_files.append(wav_path)
                results['audio_extraction']['success'] += 1
            else:
                results['audio_extraction']['failed'] += 1

        # Phase 2: Transcribe all audio
        logging.info("Phase 2: Transcribing all audio files...")
        logging.info("Loading Whisper model...")
        model = WhisperModel("distil-large-v3", device="cuda", download_root=os.getcwd(), local_files_only=True)

        srt_files = []
        for wav_path in wav_files:
            srt_path = os.path.splitext(wav_path)[0] + '.srt'
            logging.info(f"Transcribing: {wav_path}")

            if transcribe_audio(model, wav_path, srt_path):
                srt_files.append(srt_path)
                results['transcription']['success'] += 1
            else:
                results['transcription']['failed'] += 1

        # Phase 3: Extract text from all SRT files
        logging.info("Phase 3: Extracting text from all SRT files...")
        for srt_path in srt_files:
            txt_path = os.path.splitext(srt_path)[0] + '.txt'
            logging.info(f"Extracting text: {srt_path}")

            if extract_text_from_srt(srt_path, txt_path):
                results['text_extraction']['success'] += 1
            else:
                results['text_extraction']['failed'] += 1

        # Phase 4: Clean up WAV files
        logging.info("Phase 4: Cleaning up temporary WAV files...")
        for wav_path in wav_files:
            try:
                if verify_file_exists(wav_path):
                    os.remove(wav_path)
                    logging.info(f"Removed: {wav_path}")
            except Exception as e:
                logging.error(f"Error removing WAV file {wav_path}: {str(e)}")

        # Log summary
        logging.info("\nProcessing Summary:")
        for phase, counts in results.items():
            logging.info(f"{phase.replace('_', ' ').title()}:")
            logging.info(f"  Successful: {counts['success']}")
            logging.info(f"  Failed: {counts['failed']}")

    except Exception as e:
        logging.error(f"Fatal error in main process: {str(e)}")


if __name__ == "__main__":
    main()