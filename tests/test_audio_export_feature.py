"""
Test the audio export feature during experiments.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from cmvdr.experiment_manager import ExperimentManager


class TestAudioExportFeature(unittest.TestCase):
    """Test the audio export feature for saving audio files during experiments."""

    def setUp(self):
        """Create temporary directory and sample signals."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp_root_path = Path(self.temp_dir.name)
        self.fs = 16000

        # Create sample signal dictionary matching the expected structure
        signal_length = 16000  # 1 second at 16 kHz
        self.signals_dict = {
            'noisy': {'time': np.random.randn(1, signal_length)},
            'wet_rank1': {'time': np.random.randn(1, signal_length)},
            'mvdr_blind': {'time': np.random.randn(1, signal_length)},
            'cmvdr_blind': {'time': np.random.randn(1, signal_length)},
            'other_signal': {'time': np.random.randn(1, signal_length)},
        }

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_save_audio_files_creates_correct_directory_structure(self):
        """Test that audio files are saved in the correct nested directory structure."""
        parameter_to_vary = 'noise|snr_db_dir'
        param_value = 0
        idx_mtc = 0
        source_filename = 'Motor2_70.wav'

        ExperimentManager._save_audio_files_for_iteration(
            self.signals_dict, parameter_to_vary, param_value, idx_mtc, self.exp_root_path, self.fs,
            source_filename=source_filename
        )

        expected_dir = self.exp_root_path / 'audio' / parameter_to_vary / str(param_value) / '0_Motor2_70'
        self.assertTrue(expected_dir.exists(), f"Directory {expected_dir} was not created")

    def test_save_audio_files_with_prepended_order(self):
        """Test that audio files are saved with the correct prepended order numbers."""
        parameter_to_vary = 'test_param'
        param_value = 1
        idx_mtc = 0
        source_filename = 'Motor2_90'

        ExperimentManager._save_audio_files_for_iteration(
            self.signals_dict, parameter_to_vary, param_value, idx_mtc, self.exp_root_path, self.fs,
            source_filename=source_filename
        )

        expected_dir = self.exp_root_path / 'audio' / parameter_to_vary / str(param_value) / '0_Motor2_90'

        # Expected files with order numbers
        expected_files = [
            '1_noisy.wav',
            '2_wet_rank1.wav',
            '3_mvdr_blind.wav',
            '4_cmvdr_blind.wav',
            '5_other_signal.wav',  # Other signals come after the first 4
        ]

        for expected_file in expected_files:
            file_path = expected_dir / expected_file
            self.assertTrue(file_path.exists(), f"Expected file {file_path} was not created")

    def test_save_audio_files_skips_noise_signal(self):
        """Test that signals containing 'noise' in their name are skipped during export."""
        signals_with_noise = {
            **self.signals_dict,
            'noise': {'time': np.random.randn(1, 16000)},
            'noise_cov_est': {'time': np.random.randn(1, 16000)},
            'noise_reference': {'time': np.random.randn(1, 16000)},
        }

        parameter_to_vary = 'test_param'
        param_value = 1
        idx_mtc = 0

        ExperimentManager._save_audio_files_for_iteration(
            signals_with_noise, parameter_to_vary, param_value, idx_mtc, self.exp_root_path, self.fs
        )

        expected_dir = self.exp_root_path / 'audio' / parameter_to_vary / str(param_value) / str(idx_mtc)

        # Any signal containing 'noise' should not be saved
        self.assertFalse((expected_dir / 'noise.wav').exists(), "noise.wav should not be saved")
        self.assertFalse((expected_dir / 'noise_cov_est.wav').exists(), "noise_cov_est.wav should not be saved")
        self.assertFalse((expected_dir / 'noise_reference.wav').exists(), "noise_reference.wav should not be saved")

        # But other signals should be
        self.assertTrue((expected_dir / '1_noisy.wav').exists(), "noisy.wav should be saved")

    def test_save_audio_files_creates_directory_with_filename(self):
        """Test that directory is created with filename appended to MC iteration index."""
        signals_with_filename = {**self.signals_dict}

        parameter_to_vary = 'test_param'
        param_value = 1
        idx_mtc = 0
        source_filename = 'Motor2_70.wav'

        ExperimentManager._save_audio_files_for_iteration(
            signals_with_filename, parameter_to_vary, param_value, idx_mtc, self.exp_root_path, self.fs,
            source_filename=source_filename
        )

        # Directory should be named as {mc_idx}_{filename_stem}
        expected_dir = self.exp_root_path / 'audio' / parameter_to_vary / str(param_value) / '0_Motor2_70'

        self.assertTrue(expected_dir.exists(), f"Directory {expected_dir} was not created")

        # Files should still be saved inside
        self.assertTrue((expected_dir / '1_noisy.wav').exists(), "noisy.wav should be saved in the renamed directory")

    def test_save_audio_files_missing_signal_gracefully_skipped(self):
        """Test that missing signals (e.g., if 'wet' is not in dict) are gracefully skipped."""
        signals_without_wet = {
            'noisy': {'time': np.random.randn(1, 16000)},
            'mvdr_blind': {'time': np.random.randn(1, 16000)},
            'cmvdr_blind': {'time': np.random.randn(1, 16000)},
        }

        parameter_to_vary = 'test_param'
        param_value = 1
        idx_mtc = 0

        # Should not raise an error
        ExperimentManager._save_audio_files_for_iteration(
            signals_without_wet, parameter_to_vary, param_value, idx_mtc, self.exp_root_path, self.fs
        )

        expected_dir = self.exp_root_path / 'audio' / parameter_to_vary / str(param_value) / str(idx_mtc)

        # wet_rank1 should not be saved
        self.assertFalse((expected_dir / '2_wet_rank1.wav').exists(), "wet_rank1.wav should not be saved")

        # But other signals should be
        self.assertTrue((expected_dir / '1_noisy.wav').exists(), "noisy.wav should be saved")
        self.assertTrue((expected_dir / '3_mvdr_blind.wav').exists(), "mvdr_blind.wav should be saved")


if __name__ == '__main__':
    unittest.main()

