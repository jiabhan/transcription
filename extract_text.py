import os


def extract_text_from_srt(input_file, output_file):
    try:
        # Read the SRT file
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split the content into subtitle blocks
        blocks = content.strip().split('\n\n')
        text_only = []

        for block in blocks:
            lines = block.split('\n')
            # Skip the first two lines (subtitle number and timestamp)
            # Join the remaining lines which contain the actual text
            text_lines = lines[2:]
            text_only.append(' '.join(text_lines))

        # Join all extracted text with spaces
        extracted_text = ' '.join(text_only)

        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(extracted_text)

        print(f"Successfully extracted: '{input_file}' -> '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
    except Exception as e:
        print(f"Error processing '{input_file}': {str(e)}")


def process_all_srt_files():
    # Get current directory
    current_dir = os.getcwd()

    # Find all .srt files
    srt_files = [f for f in os.listdir(current_dir) if f.endswith('.srt')]

    if not srt_files:
        print("No SRT files found in the current directory!")
        return

    print(f"Found {len(srt_files)} SRT file(s)")

    # Process each SRT file
    for srt_file in srt_files:
        # Create output filename by replacing .srt with .txt
        output_file = os.path.splitext(srt_file)[0] + '.txt'
        extract_text_from_srt(srt_file, output_file)


if __name__ == "__main__":
    process_all_srt_files()
    