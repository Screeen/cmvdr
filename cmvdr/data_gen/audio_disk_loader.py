"""
Load audio files from a directory and return them as a list of numpy arrays.
"""
import os
import warnings
from pathlib import Path

import librosa
import soundfile as sf


class AudioDiskLoader:
    def __init__(self):
        """
        Initialize the AudioDiskLoader class.
        This class is used to load audio files from a specified directory.
        """
        pass

    @staticmethod
    def load_audio_files(path, fs=None) -> dict:
        """
        Load audio files from a specified directory.
        :param path: Path to the directory containing audio files or a single audio file.
        :param fs: Sampling frequency to resample the audio files. If None, uses the
            original sampling frequency of the audio files.
        :return: A dictionary files_dict of files with 'signal', 'sr', and 'path' as values and file names as keys.
        """

        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or a Path object.")

        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {path}")

        if path.is_file():
            # If the path is a file, load that single audio file
            data, sample_rate = librosa.load(path, sr=fs, mono=False)
            out = {path.name: {'signal': data, 'sr': sample_rate, 'path': path}}
            out = AudioDiskLoader.enrich_with_fileid(out)
            return out

        audio_dict = {}
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                filepath = os.path.join(path, filename)
                data, sample_rate = librosa.load(filepath, sr=fs, mono=False)
                audio_dict[filename] = {'signal': data, 'sr': sample_rate, 'path': filepath}

        if not audio_dict:
            raise ValueError("No valid audio files found in the specified directory. ")

        audio_dict = AudioDiskLoader.enrich_with_fileid(audio_dict)

        return audio_dict

    @classmethod
    def save_audio_file(cls, output_file, audio, fs):
        """
        Save a single audio file.
        :param output_file: Path to the output file.
        :param audio: Audio data to save.
        :param fs: Sampling frequency for the audio file.
        """
        sf.write(output_file, audio, fs)
        print(f"Saved audio to {output_file}")
        return output_file

    @staticmethod
    def save_audio_files(signals_dict_all_variations_time, output_folder, fs,
                         export_list=None):
        """
        Save audio files from a dictionary of signals to a specified output folder.
        :param signals_dict_all_variations_time: Dictionary containing audio signals.
        :param output_folder: Path to the output folder where audio files will be saved.
        :param fs: Sampling frequency for the audio files.
        """

        if not isinstance(signals_dict_all_variations_time, dict):
            raise ValueError("signals_dict_all_variations_time must be a dictionary.")
        if not isinstance(output_folder, (str, Path)):
            raise ValueError("output_folder must be a string or a Path object.")

        if export_list is None:
            export_list = list(signals_dict_all_variations_time.keys())

        output_folder = Path(output_folder).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        for name, signals in signals_dict_all_variations_time.items():
            for key, audio in signals.items():
                if key != 'noise_cov_est' and key in export_list:
                    # Save the audio file only if it is not noise_cov_est
                    name_no_ext = Path(name).stem  # Remove file extension
                    output_file = output_folder / f"{name_no_ext}_{key}.wav"
                    AudioDiskLoader.save_audio_file(output_file, audio, fs)

    @staticmethod
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

    @staticmethod
    def enrich_with_fileid(files_dict):
        """
        Enrich the files dictionary with file IDs.
        Args:
            files_dict (dict): Dictionary of files with 'signal', 'sr', and 'path'.
        Returns:
            dict: Enriched dictionary with file IDs.
        """
        for file_name, data in files_dict.items():

            data['fileid'] = AudioDiskLoader.find_id_from_filename(file_name)
            # Check that the file ID can be converted to integer (validation)
            try:
                if data['fileid'] is not None:
                    int(data['fileid'])  # Ensure fileid can be converted to int
            except ValueError:
                print(f"File ID {data['fileid']} for file {file_name} is not a valid integer. Setting to None.")
                data['fileid'] = None

        return files_dict
