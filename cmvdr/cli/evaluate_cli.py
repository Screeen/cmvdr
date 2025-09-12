import argparse
from cmvdr.eval.evaluate_folder import evaluate_audio_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio files in a folder.")
    parser.add_argument("-d", "--folder_denoised",
                        type=str, help="Path to the folder containing denoised audio files.")

    parser.add_argument("-r", "--folder_reference",
                        type=str, default=None,
                        help="Path to the folder containing clean reference audio files (optional). ")

    # Add optional bool flag --sort-by-snr with default False
    parser.add_argument("--sort-by-snr", action='store_true',
                        help="Sort results by SNR brackets (optional). Default is False.",
                        default=False)

    args = parser.parse_args()
    if args.folder_denoised is None:
        raise ValueError("The folder_denoised argument must be provided.")

    evaluate_audio_files(args.folder_denoised, args.folder_reference, sort_results_by_snr=args.sort_by_snr)


if __name__ == "__main__":
    main()
