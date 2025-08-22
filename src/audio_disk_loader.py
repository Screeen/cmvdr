"""
Load audio files from a directory and return them as a list of numpy arrays.
"""
import os
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
    def load_audio_files(directory, fs=None):
        """
        Load audio files from a specified directory.
        :param directory: Path to the directory containing audio files.
        :param fs: Sampling frequency to resample the audio files. If None, uses the
            original sampling frequency of the audio files.
        :return: A tuple containing a list of audio data (numpy arrays) and a list of filenames.
        """

        audio_list = []
        names = []
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                data, sample_rate = librosa.load(filepath, sr=fs, mono=False)
                audio_list.append(data)
                names.append(filename)

        return audio_list, names

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
