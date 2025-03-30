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
        logging.FileHandler('m4a_transcription.log'),
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


def convert_m4a_to_wav(m4a_path, wav_path, max_retries=3):
    """Convert M4A audio to WAV format with retry logic."""
    for attempt in range(max_retries):
        try:
            if not verify_file_exists(m4a_path):
                raise FileNotFoundError(f"M4A file not found or not accessible: {m4a_path}")

            command = [
                "ffmpeg",
                "-i", m4a_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                wav_path
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise subprocess.SubprocessError(f"FFmpeg error: {result.stderr}")

            if not verify_file_exists(wav_path):
                raise FileNotFoundError("WAV file was not created")

            return True

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} failed for {m4a_path}: {str(e)}")
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


def transcribe_m4a_file(m4a_path):
    """Transcribe a single M4A file to text."""
    try:
        # Create paths
        wav_path = os.path.splitext(m4a_path)[0] + '.wav'
        srt_path = os.path.splitext(m4a_path)[0] + '.srt'
        txt_path = os.path.splitext(m4a_path)[0] + '.txt'

        # Step 1: Convert M4A to WAV
        logging.info(f"Converting M4A to WAV: {m4a_path}")
        if not convert_m4a_to_wav(m4a_path, wav_path):
            logging.error("Failed to convert M4A to WAV")
            return False

        # Step 2: Load model and transcribe
        logging.info("Loading Whisper model...")
        model = WhisperModel("distil-large-v3", device="cuda", download_root=os.getcwd(), local_files_only=True)

        logging.info(f"Transcribing: {wav_path}")
        if not transcribe_audio(model, wav_path, srt_path):
            logging.error("Failed to transcribe audio")
            return False

        # Step 3: Extract plain text
        logging.info(f"Extracting text: {srt_path}")
        if not extract_text_from_srt(srt_path, txt_path):
            logging.error("Failed to extract text from SRT")
            return False

        # Step 4: Clean up temporary WAV file
        try:
            if verify_file_exists(wav_path):
                os.remove(wav_path)
                logging.info(f"Removed temporary file: {wav_path}")
        except Exception as e:
            logging.error(f"Error removing WAV file {wav_path}: {str(e)}")

        logging.info(f"Transcription complete. Output: {txt_path}")
        return True

    except Exception as e:
        logging.error(f"Error transcribing {m4a_path}: {str(e)}")
        return False


def main():
    try:
        print("M4A Audio Transcription Tool")
        print("----------------------------")

        # Get input file from user
        m4a_file = input("Enter the path to your M4A file: ").strip()

        if not verify_file_exists(m4a_file):
            print(f"Error: Could not access file {m4a_file}")
            return

        if not m4a_file.lower().endswith('.m4a'):
            print("Warning: The file doesn't have an .m4a extension. Proceeding anyway...")

        # Process the file
        print(f"Starting transcription of: {m4a_file}")
        success = transcribe_m4a_file(m4a_file)

        if success:
            print(f"Transcription completed successfully!")
            print(f"Output files created:")
            print(f"  - {os.path.splitext(m4a_file)[0]}.srt (with timestamps)")
            print(f"  - {os.path.splitext(m4a_file)[0]}.txt (plain text)")
        else:
            print("Transcription failed. Check the log file for details.")

    except Exception as e:
        logging.error(f"Fatal error in main process: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()