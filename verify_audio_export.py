#!/usr/bin/env python3
"""
Verification script for the audio export feature.
Demonstrates how the feature works.
"""

from pathlib import Path
from cmvdr.experiment_manager import ExperimentManager
import numpy as np
import tempfile

def verify_audio_export_feature():
    """Verify that the audio export feature is properly implemented."""

    print("=" * 70)
    print("Audio Export Feature Verification")
    print("=" * 70)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_root_path = Path(tmpdir)

        # Create sample signals
        signal_length = 16000
        signals_dict = {
            'noisy': {'time': np.random.randn(1, signal_length)},
            'wet_rank1': {'time': np.random.randn(1, signal_length)},
            'mvdr_blind': {'time': np.random.randn(1, signal_length)},
            'cmvdr_blind': {'time': np.random.randn(1, signal_length)},
            'extra_signal': {'time': np.random.randn(1, signal_length)},
            'noise_file': {'time': np.random.randn(1, signal_length)},  # Should be skipped
        }

        # Test parameters
        parameter_to_vary = 'noise|snr_db_dir'
        param_value = -10
        idx_mtc = 0
        fs = 16000
        source_filename = 'Motor2_70.wav'

        print("\n📝 Test Parameters:")
        print(f"  Parameter to vary: {parameter_to_vary}")
        print(f"  Parameter value: {param_value}")
        print(f"  Monte Carlo iteration: {idx_mtc}")
        print(f"  Source filename: {source_filename}")
        print(f"  Sampling frequency: {fs} Hz")
        print(f"  Number of signals: {len(signals_dict)}")

        # Call the save function
        print("\n💾 Saving audio files...")
        ExperimentManager._save_audio_files_for_iteration(
            signals_dict, parameter_to_vary, param_value, idx_mtc, exp_root_path, fs,
            source_filename=source_filename
        )

        # Verify results
        expected_dir = exp_root_path / 'audio' / parameter_to_vary / str(param_value) / '0_Motor2_70'

        print(f"\n✅ Verification Results:")
        print(f"  Base directory created: {expected_dir.exists()}")
        print(f"  Full path: {expected_dir}")

        # Check files
        files = sorted(expected_dir.glob('*.wav'))
        print(f"\n📂 Saved audio files ({len(files)} total):")
        for f in files:
            print(f"  ✓ {f.name}")

        # Verify that noise_file was skipped
        noise_files = list(expected_dir.glob('*noise*.wav'))
        print(f"\n🔇 Files containing 'noise' skipped:")
        if noise_files:
            print(f"  ✗ Found {len(noise_files)} files with 'noise' (should be 0!)")
            for f in noise_files:
                print(f"    - {f.name}")
        else:
            print(f"  ✓ Correctly skipped all files containing 'noise'")

        # Verify order
        expected_order = ['1_noisy.wav', '2_wet_rank1.wav', '3_mvdr_blind.wav', '4_cmvdr_blind.wav', '5_extra_signal.wav']
        actual_files = [f.name for f in sorted(expected_dir.glob('*.wav'))]

        print(f"\n🔢 File Order Verification:")
        all_correct = True
        for expected, actual in zip(expected_order, actual_files):
            match = "✓" if expected == actual else "✗"
            print(f"  {match} Expected: {expected:25} Actual: {actual}")
            if expected != actual:
                all_correct = False

        print("\n" + "=" * 70)
        if all_correct and len(files) == 5 and len(noise_files) == 0:
            print("🎉 All verification checks PASSED!")
        else:
            print("⚠️  Some checks failed")
        print("=" * 70)

if __name__ == '__main__':
    verify_audio_export_feature()



