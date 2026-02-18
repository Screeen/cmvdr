"""
Test for comparing all three coherence computation methods.
"""

import numpy as np
import unittest
from scipy.signal import ShortTimeFFT, get_window

from cmvdr.util import globs as gs
gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=42)

from cmvdr.estimation.coherence_manager import CoherenceManager
from cmvdr.estimation.modulator import Modulator


class TestCoherenceMethodComparison(unittest.TestCase):
    """Test that all three coherence methods produce comparable results."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 16000
        self.nfft = 512
        self.hop = 128
        self.win = get_window('hann', self.nfft)
        self.SFT = ShortTimeFFT(self.win, self.hop, fs=self.fs, mfft=self.nfft, 
                                scale_to='magnitude', fft_mode='twosided')
        
        # Create a test signal with multiple harmonics
        duration = 1.0
        t = np.arange(int(duration * self.fs)) / self.fs
        f0 = 200  # fundamental frequency
        self.signal_time = (np.sin(2 * np.pi * f0 * t) + 
                           0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                           0.3 * np.sin(2 * np.pi * 3 * f0 * t) +
                           0.2 * np.sin(2 * np.pi * 4 * f0 * t))
        self.signal = {'time': self.signal_time[np.newaxis, :]}
        
        # Alpha vector (must start with 0)
        self.alpha_vec_hz = np.array([0, -f0, -2*f0, -3*f0])
        
    def test_all_methods_same_shape(self):
        """Test that STFT-based methods produce output with the same shape."""
        
        # Method 1: Time-domain modulation + STFT
        max_len = self.signal['time'].shape[-1]
        modulator = Modulator(max_len, self.fs, [self.alpha_vec_hz], 
                             fast_version=True, use_filters=False, 
                             max_freq_cyclic_hz=5000)
        
        rho_time = CoherenceManager.compute_coherence(
            self.signal, self.SFT, modulator, 
            max_bin=-1, min_relative_power=1.e+3
        )
        
        # Method 2: Frequency-domain with STFT
        rho_freq_stft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=True, interpolation='none', apply_phase_correction=False
        )
        
        # Method 3: Frequency-domain with full-file DFT
        rho_freq_dft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=False, interpolation='none', apply_phase_correction=False
        )
        
        # STFT methods should have the same shape
        self.assertEqual(rho_time.shape, rho_freq_stft.shape,
                        "Time-domain and Freq-domain STFT shapes should match")
        
        # DFT will have different shape due to finer resolution, but same number of alphas
        self.assertEqual(rho_time.shape[0], rho_freq_dft.shape[0],
                        "All methods should have same number of alpha values")
        
        # Calculate frequency ranges covered
        delta_f_stft = self.SFT.delta_f
        delta_f_dft = self.fs / len(self.signal_time)
        
        freq_range_time = rho_time.shape[1] * delta_f_stft
        freq_range_stft = rho_freq_stft.shape[1] * delta_f_stft
        freq_range_dft = rho_freq_dft.shape[1] * delta_f_dft
        
        print(f"\nMethod outputs:")
        print(f"  Time-domain:  {rho_time.shape} covering {freq_range_time:.2f} Hz")
        print(f"  Freq STFT:    {rho_freq_stft.shape} covering {freq_range_stft:.2f} Hz")
        print(f"  Freq DFT:     {rho_freq_dft.shape} covering {freq_range_dft:.2f} Hz")
        
        # All methods should cover approximately the same frequency range
        self.assertAlmostEqual(freq_range_time, freq_range_stft, delta=1.0)
        self.assertAlmostEqual(freq_range_time, freq_range_dft, delta=50.0,
                              msg="DFT should cover approximately same frequency range as STFT")
        
    def test_all_methods_comparable_values(self):
        """Test that all three methods produce comparable coherence values."""
        
        # Method 1: Time-domain modulation + STFT
        max_len = self.signal['time'].shape[-1]
        modulator = Modulator(max_len, self.fs, [self.alpha_vec_hz], 
                             fast_version=True, use_filters=False, 
                             max_freq_cyclic_hz=5000)
        
        rho_time = CoherenceManager.compute_coherence(
            self.signal, self.SFT, modulator, 
            max_bin=-1, min_relative_power=1.e+3
        )
        
        # Method 2: Frequency-domain with STFT
        rho_freq_stft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=True, interpolation='linear', apply_phase_correction=True
        )
        
        # Method 3: Frequency-domain with full-file DFT
        rho_freq_dft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=False, interpolation='linear', apply_phase_correction=False
        )
        
        print("\n" + "="*70)
        print("COHERENCE METHOD COMPARISON")
        print("="*70)
        
        # Find index of alpha=0 (should be last after sorting)
        cc0_idx = len(self.alpha_vec_hz) - 1
        
        print(f"\nAlpha=0 index: {cc0_idx}")
        print(f"Alpha values (sorted): {modulator.alpha_vec_hz_}")
        
        # Check alpha=0 coherence (should be 1.0 for all methods)
        print(f"\nAlpha=0 coherence (should be ~1.0):")
        print(f"  Time-domain:     {rho_time[cc0_idx, 10]:.6f}")
        print(f"  Freq-domain STFT: {rho_freq_stft[cc0_idx, 10]:.6f}")
        # For DFT, use equivalent bin (scale by resolution difference)
        dft_bin_idx = int(10 * rho_freq_dft.shape[1] / rho_time.shape[1])
        print(f"  Freq-domain DFT:  {rho_freq_dft[cc0_idx, dft_bin_idx]:.6f}")
        
        self.assertAlmostEqual(rho_time[cc0_idx, 10], 1.0, places=5)
        self.assertAlmostEqual(rho_freq_stft[cc0_idx, 10], 1.0, places=5)
        self.assertAlmostEqual(rho_freq_dft[cc0_idx, dft_bin_idx], 1.0, places=5)
        
        # Compare overall statistics
        print(f"\nOverall statistics:")
        print(f"  Time-domain:     mean={rho_time.mean():.4f}, std={rho_time.std():.4f}")
        print(f"  Freq-domain STFT: mean={rho_freq_stft.mean():.4f}, std={rho_freq_stft.std():.4f}")
        print(f"  Freq-domain DFT:  mean={rho_freq_dft.mean():.4f}, std={rho_freq_dft.std():.4f}")
        
        # Correlation between STFT methods (should be high since they have same shape)
        corr_time_stft = np.corrcoef(rho_time.flatten(), rho_freq_stft.flatten())[0, 1]
        
        print(f"\nCorrelations:")
        print(f"  Time vs Freq-STFT: {corr_time_stft:.4f}")
        
        # Correlation should be reasonably high
        self.assertGreater(corr_time_stft, 0.5, 
                          "Time-domain and Freq-domain STFT should be reasonably correlated")
        
        # Check that all values are in valid range [0, 1]
        self.assertTrue(np.all(rho_time >= -0.01))
        self.assertTrue(np.all(rho_time <= 1.01))
        self.assertTrue(np.all(rho_freq_stft >= -0.01))
        self.assertTrue(np.all(rho_freq_stft <= 1.01))
        self.assertTrue(np.all(rho_freq_dft >= -0.01))
        self.assertTrue(np.all(rho_freq_dft <= 1.01))
        
    def test_high_res_stft_produces_larger_output(self):
        """Test that high-resolution STFT produces more frequency bins."""
        
        # Standard resolution STFT
        rho_standard = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=True, interpolation='none', apply_phase_correction=False
        )
        
        # High-resolution STFT (4x resolution)
        nfft_high = 2048
        win_high = get_window('hann', nfft_high)
        SFT_high = ShortTimeFFT(win_high, self.hop, fs=self.fs, mfft=nfft_high,
                               scale_to='magnitude', fft_mode='twosided')
        
        rho_high = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, SFT_high, self.alpha_vec_hz,
            max_bin=-1, min_relative_power=1.e+3,
            use_stft=True, interpolation='none', apply_phase_correction=False
        )
        
        print(f"\n" + "="*70)
        print("HIGH-RES STFT TEST")
        print("="*70)
        print(f"Standard STFT: nfft={self.nfft}, delta_f={self.SFT.delta_f:.2f} Hz")
        print(f"  Output shape: {rho_standard.shape}")
        print(f"High-res STFT: nfft={nfft_high}, delta_f={SFT_high.delta_f:.2f} Hz")
        print(f"  Output shape: {rho_high.shape}")
        
        # High-res should have more bins
        self.assertGreater(rho_high.shape[1], rho_standard.shape[1],
                          "High-resolution STFT should produce more frequency bins")
        
        # The ratio should be approximately nfft_high / self.nfft
        ratio = rho_high.shape[1] / rho_standard.shape[1]
        expected_ratio = nfft_high / self.nfft
        print(f"\nBin count ratio: {ratio:.2f} (expected ~{expected_ratio:.2f})")
        
        # Should be within 20% of expected ratio
        self.assertGreater(ratio, expected_ratio * 0.8)
        self.assertLess(ratio, expected_ratio * 1.2)


if __name__ == '__main__':
    unittest.main()
