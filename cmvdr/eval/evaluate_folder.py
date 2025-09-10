"""
Script to evaluate all audio files in a folder.
Usage: python evaluate_folder.py <folder-denoised> --folder-reference <folder-reference>
Example:
    ../../dns_challenge/datasets/dev/debug --folder-reference ../../dns_challenge/datasets/dev/debug

Metrics:
    Specified in eval_config.yaml.

Output:
    Pretty print + rich.table to terminal
"""

import argparse
import warnings
from pathlib import Path

import yaml
import librosa
from rich.console import Console
from rich.table import Table
import numpy as np
from tqdm import tqdm
import cmvdr.eval.metrics_manager as metrics_manager


def load_audio_files(folder, sr=None):
    """
    Load audio files from the specified folder.
    Args:
        folder (str or Path): Path to the folder containing audio files.
        sr (int, optional): Sample rate to load the audio files. If None, uses the default sample rate.
    Returns:
        dict: A dictionary where keys are file names and values are dictionaries with 'signal', 'sr', and 'path'.
    """
    # Implement the logic to load audio files from the folder
    print(f"Loading audio files from: {folder}")

    # use librosa to load audio files
    audio_files = list(Path(folder).glob("*.wav"))

    loaded_files = {}
    for audio_file in audio_files:
        try:
            signal, sr = librosa.load(audio_file, sr=sr)
            loaded_files[audio_file.name] = {
                'signal': signal,
                'sr': sr,
                'path': audio_file
            }
            print(f"Loaded {audio_file.name} with sample rate {sr} Hz")
        except Exception as e:
            print(f"Error loading {audio_file.name}: {e}")

    return loaded_files


def evaluate_individual_files(denoised_files, metrics, reference_files=None):
    """
    Evaluate individual audio files using specified metrics.
    Args:
        denoised_files (dict): Dictionary of denoised audio files with 'signal', 'sr', and 'path'.
        metrics (list): List of metrics to compute (e.g., ['STOI', 'PESQ', 'DNSMOS', 'SI-SDR']).
        reference_files (dict, optional): Dictionary of reference audio files with 'signal' and 'sr'.
    Returns:
        dict: A dictionary with file names as keys and computed metrics as values.
    """
    if not denoised_files:
        raise ValueError("No denoised files provided for evaluation.")
    if not metrics:
        raise ValueError("No metrics specified for evaluation.")

    mm = metrics_manager.MetricsManager()

    results_dict = {}
    for metric in metrics:
        print(f"Computing metric: {metric}")
        for file_name, data in tqdm(denoised_files.items(), total=len(denoised_files)):
            signal = data['signal']
            sr = data['sr']

            fileid = data["fileid"]
            if reference_files and fileid in (r["fileid"] for r in reference_files.values()):
                ref_signal = next(r["signal"] for r in reference_files.values() if r["fileid"] == fileid)
            else:
                ref_signal = data["signal"]
                warnings.warn(f"No reference for {file_name}, using denoised signal as reference.")

            signal = signal[:ref_signal.shape[0]]

            if metric == 'STOI':
                stoi_value = mm.compute_stoi(ref_signal, signal, sr)
                results_dict[file_name] = results_dict.get(file_name, {})
                results_dict[file_name]['STOI'] = stoi_value
                tqdm.write(f"Computed STOI for {file_name}: {stoi_value:.4f}")

            elif metric == 'PESQ':
                pesq_value = mm.compute_pesq(ref_signal, signal, sr)
                results_dict[file_name] = results_dict.get(file_name, {})
                results_dict[file_name]['PESQ'] = pesq_value
                tqdm.write(f"Computed PESQ for {file_name}: {pesq_value:.4f}")

            elif metric == 'DNSMOS':
                dnsmos_value = mm.compute_dnsmos(signal, sr)
                results_dict[file_name] = results_dict.get(file_name, {})
                results_dict[file_name]['DNSMOS'] = dnsmos_value
                tqdm.write(f"Computed DNSMOS for {file_name}: {dnsmos_value:.4f}")

            elif metric == 'SI-SDR':
                si_sdr_value = mm.sisdr(ref_signal, signal)
                results_dict[file_name] = results_dict.get(file_name, {})
                results_dict[file_name]['SI-SDR'] = si_sdr_value
                tqdm.write(f"Computed SI-SDR for {file_name}: {si_sdr_value:.4f}")

            else:
                warnings.warn(f"Metric {metric} is not recognized. Skipping.")

    return results_dict


def normalize_volume(x_samples, max_value=0.9):
    if np.max(np.abs(x_samples)) < 1e-6:
        warnings.warn(f"Skipping normalization as it would amplify numerical noise.")
        return x_samples
    else:
        return max_value * x_samples / np.max(np.abs(x_samples))


def split_results_by_snr(results_dict):
    # Based on the SNR field in results_dict, split the results into several dictionaries, one per SNR bracket
    # Each bracket is centered around a multiple of 5 dB: -20, -15, -10, -5, 0, 5, 10, 15, 20
    snr_brackets = {}
    for file_name, data in results_dict.items():
        snr = data.get('SNR', None)
        if snr is None:
            warnings.warn(f"No SNR found for {file_name}, skipping SNR-based splitting.")
            continue

        snr_bracket = 5 * round(snr / 5)
        if snr_bracket not in snr_brackets:
            snr_brackets[snr_bracket] = {}

        snr_brackets[snr_bracket][file_name] = data

    return snr_brackets


def evaluate_audio_files(folder_denoised, folder_reference=None, sort_results_by_snr=False):
    """
    Evaluate audio files in a specified folder.
    Args:
        folder_denoised (str or Path): Path to the folder containing denoised audio
        folder_reference (str or Path, optional): Path to the folder containing reference audio files.
    Returns:
        None: Prints evaluation results to the console.
    """

    # Load configuration file located in same directory as this script
    # This can be used to load evaluation parameters, metrics, etc.
    path_to_config = Path(__file__).parent / 'eval_config.yaml'
    if not path_to_config.exists():
        raise FileNotFoundError(f"Configuration file {path_to_config} does not exist.")

    cfg = load_configuration_from_path(path_to_config)
    sr = cfg.get('fs', 0)

    # Load audio files, compute metrics, and save results
    folder_denoised = Path(folder_denoised)
    if not folder_denoised.exists():
        raise FileNotFoundError(f"The folder {folder_denoised} does not exist.")
    print(f"Evaluating audio files in: {folder_denoised}")

    denoised_files = load_audio_files(folder_denoised, sr)
    if not denoised_files:
        print("No audio files found to evaluate.")
        return

    # Load reference files if provided
    reference_files = {}
    if folder_reference:
        print(f"Using reference folder: {folder_reference}")
        reference_files = load_audio_files(folder_reference, sr)
        if not reference_files:
            print("No reference audio files found.")
            return

    # Normalize volumes of denoised and reference files
    for file_name, data in denoised_files.items():
        data['signal'] = normalize_volume(data['signal'])
    if reference_files:
        for file_name, data in reference_files.items():
            data['signal'] = normalize_volume(data['signal'])

    # Placeholder for evaluation logic
    # Metrics to compute are specified in the configuration file
    print(f"Loaded {len(denoised_files)} audio files for evaluation.")

    denoised_files = enrich_with_fileid(denoised_files)
    reference_files = enrich_with_fileid(reference_files)

    metrics = cfg.get('metrics', [])
    results_dict = evaluate_individual_files(denoised_files, metrics, reference_files if folder_reference else None)

    if sort_results_by_snr:
        # Enrich results with SNR information if available in filenames
        # Next, divide the results by SNR if SNR information is available. This allows to compute statistics per SNR.
        results_dict = enrich_with_snr(results_dict)
        results_by_snr = split_results_by_snr(results_dict)
        for snr, res in sorted(results_by_snr.items()):
            stats = summarize_statistics(res)
            print_as_rich_table(stats, title=f"Results around SNR {snr} dB")

    results_stats = summarize_statistics(results_dict)
    print_as_rich_table(results_stats, title="Overall results")

    # Save results to a file
    # save_results(results_dict, folder)


def summarize_statistics(results_dict, skip_metrics=None) -> dict:
    """ Summarize statistics of results_dict """

    # Assuming that all files use the same metrics
    if not results_dict:
        warnings.warn("No results to summarize.")
        return {}

    if skip_metrics is None:
        skip_metrics = ['SNR', 'fileid']

    metrics = list(next(iter(results_dict.values())).keys())
    results_stats = {}
    for metric in metrics:
        if metric in skip_metrics:
            continue

        # Collect all values for this metric
        metric_values = np.array([results_dict[file_name].get(metric, 0) for file_name in results_dict.keys()])

        if np.any(metric_values):
            mean_value = np.mean(metric_values)
            stderr_value = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values)) if len(metric_values) > 1 else 0
            results_stats[metric] = {
                'mean': mean_value,
                'stderr': stderr_value
            }

    return results_stats


def print_as_rich_table(results_stats, title=''):
    """ Print results statistics as a rich table """

    console = Console()
    table = Table(title=title)

    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("StdErr", justify="right", style="yellow")

    for metric, stats in results_stats.items():
        table.add_row(metric, f"{stats['mean']:.4f}", f"{stats['stderr']:.4f}")

    console.print(table)


def load_configuration_from_path(configuration_path):
    """ Read settings from configuration file """

    with open(configuration_path, 'r') as f:
        conf_dict = yaml.safe_load(f)
    print(f"Loaded configuration file {configuration_path}")
    return conf_dict


def enrich_with_fileid(files_dict):
    """
    Enrich the files dictionary with file IDs.
    Args:
        files_dict (dict): Dictionary of files with 'signal', 'sr', and 'path'.
    Returns:
        dict: Enriched dictionary with file IDs.
    """
    for file_name, data in files_dict.items():

        data['fileid'] = find_id_from_filename(file_name)
        # Check that the file ID can be converted to integer (validation)
        try:
            if data['fileid'] is not None:
                int(data['fileid'])  # Ensure fileid can be converted to int
        except ValueError:
            warnings.warn(f"File ID {data['fileid']} for file {file_name} is not a valid integer. Setting to None.")
            data['fileid'] = None

    return files_dict


def enrich_with_snr(files_dict):
    for file_name, data in files_dict.items():
        data['SNR'] = find_snr_from_filename(file_name)
        # Check that the SNR can be converted to integer (validation)
        try:
            if data['SNR'] is not None:
                int(data['SNR'])  # Ensure fileid can be converted to int
        except ValueError:
            warnings.warn(f"SNR {data['SNR']} for file {file_name} is not a valid integer. Setting to None.")
            data['SNR'] = None

    return files_dict


def find_id_from_filename(file_name):
    """
    Extract file ID from the file name.
    Args:
        file_name (str): Name of the file. Example: noisy_fileid_9_book_03576_chp_0017_reader_02708_7_noise_004237_snr-4_tl-16.wav
    Returns:
        str: Extracted file ID.
    """

    # Assuming the file name always contains *fileid_FILEID*
    # Example: "mixture_001_fileid_12345.wav"
    parts = Path(file_name).stem.split('_')
    for (idx, part) in enumerate(parts):
        if part == "fileid" and (idx + 1) < len(parts):
            return parts[idx + 1]
    warnings.warn(f"No file ID found in {file_name}. Returning None.")
    return None  # Return None if no file ID is found


def find_snr_from_filename(file_name):
    """
    Extract SNR from the file name.
    Args:
        file_name (str): Name of the file.
        Example 1: noisy_fileid_9_book_03576_chp_0017_reader_02708_7_noise_004237_snr-4_tl-16.wav
        Example 2: noisy_fileid_9_book_03576_chp_0017_reader_02708_7_noise_004237_snr4_tl-16.wav
    Returns:
        str: Extracted SNR.
    """
    parts = Path(file_name).stem.split('_')
    for (idx, part) in enumerate(parts):
        if part.startswith("snr"):
            snr_str = part[3:]
            if snr_str.startswith('-') or snr_str.isdigit():
                return int(snr_str)

    warnings.warn(f"No SNR found in {file_name}. Returning None.")
    return None  # Return None if no SNR is found
