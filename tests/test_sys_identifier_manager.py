import unittest
from pathlib import Path

import numpy as np
import sys
import os

import src.utils as u
import src.globs as gs
from src.manager import Manager
gs.rng, _ = gs.compute_rng(seed_is_random=True)


class TestSysIdentifierManager(unittest.TestCase):

    @staticmethod
    def load_vowel_recording(N_num_samples, fs_, offset_=0, selected_people=(), smoothing_window=True):
        """
        Load a random signal from the North Texas Vowel Database.

        References
        - Assmann, P., & Katz, W. (2005). Synthesis fidelity and time-varying spectral change in vowels. Journal of the Acoustical Society of America, 117, 886-895.
        - Katz, W., & Assmann, P. (2001). Identification of children’s and adults’ vowels: Intrinsic fundamental frequency, fundamental frequency dynamics, and presence of voicing. Journal of Phonetics, 29, 23-51.
        - Assmann, P., & Katz, W. (2000) Time-varying spectral change in the vowels of children and adults. Journal of the Acoustical Society of America, 108, 1856-1866.

        URL
        https://labs.utdallas.edu/speech-production-lab/links/utd-nt-vowel-database/

        """
        if N_num_samples < 1:
            raise ValueError(f"{N_num_samples = } must be a positive integer.")

        module_parent = Path(__file__).resolve().parent.parent
        dataset_path_parent = module_parent.parent / 'datasets' / 'north_texas_vowels'

        if not dataset_path_parent.exists():
            raise ValueError(
                f"Path {dataset_path_parent} does not exist. Download the dataset from the URL in the docstring.")

        labels_path = dataset_path_parent / 'labels.csv'
        data_path = dataset_path_parent / 'data'

        # Load labels from CSV file
        labels = np.loadtxt(labels_path, delimiter=';', skiprows=1, dtype=str)

        # Keep only labels whose first column is a valid file name (ends with two numbers)
        labels = labels[np.array([x[-2:].isdigit() for x in labels[:, 0]])]

        # Column 0 is the file name
        # Column 1 is adult or child (adult males are labeled as '1', adult females as '2', children as '3', '4' and '5')
        # Filter to keep only children (3 4 and 5)
        if not selected_people:
            # selected_people = ['3', '4', '5']
            selected_people = ['1', '2']
            # selected_people = ['1']

        labels_filtered = labels[np.isin(labels[:, 1], selected_people)]

        # Choose random sample
        random_file_name = lambda: data_path / (labels_filtered[gs.rng.choice(labels_filtered.shape[0]), 0] + '.wav')

        counter = 0
        file_path = data_path / random_file_name()
        while not file_path.exists() and counter < 10:
            file_path = data_path / random_file_name()
            counter += 1

        s, fs_ = u.load_audio_file_random_offset(file_path, fs_, N_num_samples)

        return s

    def setUp(self):
        self.manager = Manager()

    def test_load_signal_with_valid_input(self):
        signal = self.load_vowel_recording(1000, 44100)
        self.assertEqual(len(signal), 1000)

    def test_load_signal_with_zero_samples(self):
        with self.assertRaises(ValueError):
            self.load_vowel_recording(0, 44100)

    def test_load_signal_with_negative_samples(self):
        with self.assertRaises(ValueError):
            self.load_vowel_recording(-1000, 44100)

    def test_choose_loud_bins_with_valid_input(self):
        S_pow = np.array([1, 2, 3, 4, 5])
        result = self.manager.choose_loud_bins(S_pow, power_ratio_quiet_definition=2)

        # Should contain bins which are louder than half of the maximum power: 3, 4, 5
        expected_result = np.array([2, 3, 4])
        self.assertTrue(np.array_equal(result, expected_result), "The result does not match the expected output.")


if __name__ == '__main__':
    unittest.main()
