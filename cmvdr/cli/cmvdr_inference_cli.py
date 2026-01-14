import os
from contextlib import redirect_stdout
from pathlib import Path
import argparse
import tempfile
import shutil
import multiprocessing as mp
from typing import List, Tuple, Dict
from cmvdr.util import config, globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)
from cmvdr import experiment_manager


def get_audio_files(folder: Path) -> List[Path]:
    """Get all audio files from a folder."""
    extensions = ['.wav', '.flac', '.mp3', '.ogg']
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    return sorted(files)


def extract_fileid(filename: str) -> str:
    """
    Extract fileid from filename.
    For noisy files: noisy_fileid_0_clean_fileid_11368_noise_004242_snr-9_tl-17.wav -> '0'
    For noise files: noise_fileid_10490.wav -> '10490'
    Returns the fileid that appears after the first '_fileid_'
    """
    stem = Path(filename).stem
    parts = stem.split('_fileid_')
    if len(parts) > 1:
        # Get the part after '_fileid_' and extract just the numeric ID
        # Split by '_' and take the first element
        fileid_part = parts[1].split('_')[0]
        return fileid_part
    return stem


def create_batch_symlinks(files: List[Path], batch_dir: Path) -> None:
    """Create symlinks for a batch of files."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        symlink = batch_dir / file.name
        if not symlink.exists():
            symlink.symlink_to(file.resolve())


def match_noise_files(input_files: List[Path], noise_base: Path) -> Dict[Path, Path]:
    """
    Match input files to their corresponding noise files.
    Following the convention: append _fileid_123.wav to both noise and noisy files.
    """
    if not noise_base or not noise_base.exists():
        return {}

    noise_files_all = get_audio_files(noise_base)

    # Build a mapping of fileid -> noise file path
    noise_by_fileid = {}
    for noise_file in noise_files_all:
        fileid = extract_fileid(noise_file.name)
        noise_by_fileid[fileid] = noise_file

    # Match input files to noise files
    matched = {}
    for input_file in input_files:
        fileid = extract_fileid(input_file.name)
        if fileid in noise_by_fileid:
            matched[input_file] = noise_by_fileid[fileid]

    return matched


def process_batch(args: Tuple[int, List[Path], Path, Path, Path, dict, bool]) -> Tuple[int, bool, str, int]:
    """Process a single batch of files."""
    batch_idx, files, input_base, output_base, noise_base, cfg, verbose = args

    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory(prefix=f'cmvdr_batch_{batch_idx}_') as temp_dir:
            temp_path = Path(temp_dir)
            batch_input_dir = temp_path / "input"
            batch_output_dir = temp_path / "output"

            # Create symlinks for input files
            create_batch_symlinks(files, batch_input_dir)

            # Create symlinks for noise files if provided
            batch_noise_dir = None
            if noise_base:
                # Match noise files to input files based on fileid
                noise_matches = match_noise_files(files, noise_base)

                if noise_matches:
                    batch_noise_dir = temp_path / "noise"
                    batch_noise_dir.mkdir(parents=True, exist_ok=True)

                    for input_file, noise_file in noise_matches.items():
                        # Create symlink with same name as input file for proper matching
                        symlink = batch_noise_dir / input_file.name
                        if not symlink.exists():
                            symlink.symlink_to(noise_file.resolve())

            # Run inference on this batch
            em = experiment_manager.ExperimentManager()
            kwargs = {
                'input_path': batch_input_dir,
                'noise_path': batch_noise_dir,
                'output_path': batch_output_dir,
                'cfg': cfg,
                'verbose': verbose
            }

            if verbose:
                print(f"Processing batch {batch_idx} ({len(files)} files)...")
                em.run_cmvdr_inference_folder(**kwargs)
            else:
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    em.run_cmvdr_inference_folder(**kwargs)

            # Copy output files to final destination
            files_processed = 0
            if batch_output_dir.exists():
                for output_file in batch_output_dir.rglob('*'):
                    if output_file.is_file():
                        relative_path = output_file.relative_to(batch_output_dir)
                        final_output = output_base / relative_path
                        final_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(output_file, final_output)
                        files_processed += 1

        return batch_idx, True, f"Batch {batch_idx} completed successfully", files_processed

    except Exception as e:
        return batch_idx, False, f"Batch {batch_idx} failed: {str(e)}", 0


def run_parallel_inference(input_path: Path, output_path: Path, noise_path: Path,
                           cfg: dict, verbose: bool, num_workers: int, batch_size: int):
    """Run cMVDR inference in parallel on batches of files."""

    # Get all audio files
    if input_path.is_file():
        print("Parallel processing requires a folder input, not a single file.")
        print("Use the standard mode (without -p flag) for single file processing.")
        return

    all_files = get_audio_files(input_path)
    total_files = len(all_files)

    if total_files == 0:
        print("No audio files found in the input folder.")
        return

    print(f"Found {total_files} audio files")
    print(f"Processing with {num_workers} workers in batches of {batch_size} files")

    # Create output directory
    if output_path is None:
        output_path = input_path.with_name(input_path.name + '_cmvdr')
    output_path.mkdir(parents=True, exist_ok=True)

    # Split files into batches
    batches = []
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i + batch_size]
        batches.append((i // batch_size, batch_files, input_path, output_path, noise_path, cfg, verbose))

    print(f"Created {len(batches)} batches")

    # Process batches in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_batch, batches)

    # Report results
    successful = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - successful
    total_processed = sum(count for _, _, _, count in results)

    print(f"\nProcessing complete:")
    print(f"  Successful batches: {successful}/{len(batches)}")
    print(f"  Failed batches: {failed}/{len(batches)}")
    print(f"  Total files processed: {total_processed}/{total_files}")

    if failed > 0:
        print("\nFailed batches:")
        for batch_idx, success, msg, _ in results:
            if not success:
                print(f"  {msg}")


def main():
    cfg_original = config.load_configuration_outer('inference')
    cfg_original = config.assign_default_values(cfg_original)
    cfg_original['beamforming']['methods'] = ['cmvdr_blind']

    parser = argparse.ArgumentParser(description="Run cMVDR inference on a single file or a folder of audio files.")
    parser.add_argument('-i', '--input_path', required=True, type=str, help="Path to the input audio file or folder.")
    parser.add_argument("-o", "--output_path", type=str, default=None,
                        help="Path to the output folder. If not provided, output will be saved in the same folder as input.")
    parser.add_argument("-n", "--noise_path", type=str, default=None,
                        help="Path to the noise audio file or folder (optional, to estimate noise frequency). "
                             "To match input files, append _fileid_123.wav to the noise and the noisy files.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="If set, print detailed logs to the console.")
    parser.add_argument("-p", "--parallel", action="store_true", default=False,
                        help="If set, process files in parallel using multiple workers.")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores).")
    parser.add_argument("-b", "--batch_size", type=int, default=100,
                        help="Number of files to process per batch in parallel mode (default: 100).")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    noise_path = Path(args.noise_path).expanduser().resolve() if args.noise_path else None

    if args.parallel:
        # Parallel processing mode
        num_workers = args.workers if args.workers else mp.cpu_count()
        run_parallel_inference(input_path, output_path, noise_path,
                               cfg_original, args.verbose, num_workers, args.batch_size)
    else:
        # Original sequential processing
        em = experiment_manager.ExperimentManager()
        kwargs = {'input_path': input_path, 'noise_path': noise_path, 'output_path': output_path,
                  'cfg': cfg_original, 'verbose': args.verbose}

        if args.verbose:
            em.run_cmvdr_inference_folder(**kwargs)
        else:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                em.run_cmvdr_inference_folder(**kwargs)


if __name__ == "__main__":
    main()