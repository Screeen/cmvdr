#!/usr/bin/env python3
"""
Trim WAV files by removing specified seconds from start and end.
Preserves multichannel, sample rate, and all other audio properties.
"""

import argparse
from pathlib import Path
import soundfile as sf


def trim_wav(input_path, output_path, trim_start, trim_end):
    """
    Trim a WAV file by removing seconds from start and end.

    Args:
        input_path: Path to input WAV file
        output_path: Path to output WAV file
        trim_start: Seconds to remove from start
        trim_end: Seconds to remove from end
    """
    # Read the audio file
    data, samplerate = sf.read(input_path)

    # Calculate samples to trim
    start_samples = int(trim_start * samplerate)
    end_samples = int(trim_end * samplerate)

    # Calculate total length
    total_samples = len(data)

    # Check if trimming is valid
    if start_samples + end_samples >= total_samples:
        print(
            f"Warning: {input_path.name} is too short to trim (duration: {total_samples / samplerate:.2f}s). Skipping.")
        return False

    # Trim the audio
    trimmed_data = data[start_samples:total_samples - end_samples]

    # Write the trimmed audio, preserving all properties
    sf.write(output_path, trimmed_data, samplerate, subtype=sf.info(input_path).subtype)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Trim WAV files by removing specified seconds from start and end.'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Folder containing WAV files to process'
    )
    parser.add_argument(
        '--trim-start',
        type=float,
        default=7.0,
        help='Seconds to remove from start of each file (default: 7.0)'
    )
    parser.add_argument(
        '--trim-end',
        type=float,
        default=10.0,
        help='Seconds to remove from end of each file (default: 10.0)'
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        default='trimmed',
        help='Output subfolder name (default: trimmed)'
    )

    args = parser.parse_args()

    # Setup paths
    input_folder = Path(args.input_folder)
    output_folder = input_folder / args.output_folder

    # Check if input folder exists
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)

    # Find all WAV files
    wav_files = list(input_folder.glob('*.wav')) + list(input_folder.glob('*.WAV'))

    if not wav_files:
        print(f"No WAV files found in '{input_folder}'")
        return

    print(f"Found {len(wav_files)} WAV file(s)")
    print(f"Trimming: {args.trim_start}s from start, {args.trim_end}s from end")
    print(f"Output folder: {output_folder}")
    print("-" * 60)

    # Process each WAV file
    processed = 0
    for wav_file in wav_files:
        output_path = output_folder / wav_file.name
        print(f"Processing: {wav_file.name}... ", end='', flush=True)

        try:
            if trim_wav(wav_file, output_path, args.trim_start, args.trim_end):
                print("✓")
                processed += 1
            else:
                print("✗ (skipped)")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("-" * 60)
    print(f"Successfully processed {processed}/{len(wav_files)} files")


if __name__ == '__main__':
    main()