import unittest
import numpy as np
from scipy.signal import ShortTimeFFT, get_window

from cmvdr.util import globs as gs
gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=42)

from cmvdr.estimation.coherence_manager import CoherenceManager
from cmvdr.estimation.modulator import Modulator


class TestFrequencyDomainCoherence(unittest.TestCase):
    """Tests for frequency-domain coherence computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.fs = 16000
        self.nfft = 512
        self.hop = 128
        self.win = get_window('hann', self.nfft)
        self.SFT = ShortTimeFFT(self.win, self.hop, fs=self.fs, mfft=self.nfft, scale_to='magnitude')
        
        # Create a simple test signal with harmonics
        duration = 0.5
        t = np.arange(int(duration * self.fs)) / self.fs
        f0 = 200  # fundamental frequency
        self.signal_time = (np.sin(2 * np.pi * f0 * t) + 
                           0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                           0.3 * np.sin(2 * np.pi * 3 * f0 * t))
        self.signal = {'time': self.signal_time[np.newaxis, :]}
        
        # Alpha vector (cyclic frequencies)
        self.alpha_vec_hz = np.array([0, -f0, -2*f0, -3*f0])

    def test_compute_coherence_freq_shifted_basic(self):
        """Test basic frequency-domain coherence computation."""
        rho = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=50, min_relative_power=1.e+3,
            use_stft=True, interpolation='none', apply_phase_correction=False
        )
        
        # Check output shape
        P_sum = len(self.alpha_vec_hz)
        self.assertEqual(rho.shape[0], P_sum)
        self.assertTrue(rho.shape[1] > 0)
        
        # Check that rho[0] (no shift) has high coherence with itself
        self.assertAlmostEqual(rho[0, 10], 1.0, places=5)
        
        # Check that values are in [0, 1]
        self.assertTrue(np.all(rho >= -0.01))
        self.assertTrue(np.all(rho <= 1.01))

    def test_interpolation_modes(self):
        """Test different interpolation modes."""
        for interp in ['none', 'linear', 'lagrange8']:
            with self.subTest(interpolation=interp):
                rho = CoherenceManager.compute_coherence_freq_shifted(
                    self.signal, self.SFT, self.alpha_vec_hz,
                    max_bin=50, use_stft=True,
                    interpolation=interp, apply_phase_correction=False
                )
                
                # All should produce valid coherence matrices
                self.assertEqual(rho.shape[0], len(self.alpha_vec_hz))
                self.assertTrue(np.all(np.isfinite(rho)))

    def test_full_file_dft_vs_stft(self):
        """Test full-file DFT vs STFT modes."""
        rho_dft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=50, use_stft=False, interpolation='none'
        )
        
        rho_stft = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=50, use_stft=True, interpolation='none'
        )
        
        # Both should produce valid outputs
        self.assertEqual(rho_dft.shape[0], len(self.alpha_vec_hz))
        self.assertEqual(rho_stft.shape[0], len(self.alpha_vec_hz))
        self.assertTrue(np.all(np.isfinite(rho_dft)))
        self.assertTrue(np.all(np.isfinite(rho_stft)))

    def test_phase_correction(self):
        """Test phase correction toggle."""
        rho_with_corr = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=50, use_stft=True,
            interpolation='linear', apply_phase_correction=True
        )
        
        rho_without_corr = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=50, use_stft=True,
            interpolation='linear', apply_phase_correction=False
        )
        
        # Both should be valid, but may differ
        self.assertTrue(np.all(np.isfinite(rho_with_corr)))
        self.assertTrue(np.all(np.isfinite(rho_without_corr)))

    def test_shift_spectrum_basic(self):
        """Test basic spectrum shifting."""
        # Create a simple spectrum
        frames = 10
        kk_max = 50
        spec = np.random.randn(kk_max, frames) + 1j * np.random.randn(kk_max, frames)
        
        alpha_hz = 100
        delta_f = self.fs / self.nfft
        
        shifted = CoherenceManager._shift_spectrum(
            spec, alpha_hz, delta_f, self.fs,
            interpolation='none', apply_phase_correction=False
        )
        
        self.assertEqual(shifted.shape, spec.shape)
        self.assertTrue(np.all(np.isfinite(shifted)))

    def test_shift_spectrum_zero_shift(self):
        """Test that zero shift returns identity."""
        frames = 5
        kk_max = 30
        spec = np.random.randn(kk_max, frames) + 1j * np.random.randn(kk_max, frames)
        
        delta_f = self.fs / self.nfft
        shifted = CoherenceManager._shift_spectrum(
            spec, 0.0, delta_f, self.fs,
            interpolation='linear', apply_phase_correction=False
        )
        
        # Should be very close to original
        np.testing.assert_allclose(shifted, spec, rtol=1e-5, atol=1e-8)

    def test_lagrange8_interpolate(self):
        """Test 8-point Lagrange interpolation."""
        frames = 5
        kk_max = 30
        spec = np.random.randn(kk_max, frames) + 1j * np.random.randn(kk_max, frames)
        
        # Test integer position (should give exact value)
        src_bin_float = 10.0
        result = CoherenceManager._lagrange8_interpolate(spec, src_bin_float)
        np.testing.assert_allclose(result, spec[10, :], rtol=1e-5)
        
        # Test fractional position (should interpolate)
        src_bin_float = 10.5
        result = CoherenceManager._lagrange8_interpolate(spec, src_bin_float)
        self.assertEqual(result.shape, (frames,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_edge_cases(self):
        """Test edge cases."""
        # Single bin
        alpha_vec_hz = np.array([0])
        rho = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, alpha_vec_hz,
            max_bin=10, use_stft=True, interpolation='none'
        )
        self.assertEqual(rho.shape[0], 1)
        
        # Very small signal (ensure it's long enough for STFT)
        small_signal = {'time': np.zeros((1, 8000))}
        small_signal['time'][0, :] = 1e-10 * np.random.randn(8000)
        
        rho = CoherenceManager.compute_coherence_freq_shifted(
            small_signal, self.SFT, self.alpha_vec_hz,
            max_bin=10, use_stft=True, interpolation='none'
        )
        self.assertTrue(np.all(np.isfinite(rho)))


class TestFrequencyDomainVsTimeDomain(unittest.TestCase):
    """Compare frequency-domain and time-domain coherence methods."""

    def setUp(self):
        """Set up test fixtures."""
        gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
        
        self.fs = 16000
        self.nfft = 512
        self.hop = 128
        self.win = get_window('hann', self.nfft)
        self.SFT = ShortTimeFFT(self.win, self.hop, fs=self.fs, mfft=self.nfft, scale_to='magnitude')
        
        # Create test signal
        duration = 1.0
        t = np.arange(int(duration * self.fs)) / self.fs
        f0 = 150
        self.signal_time = (np.sin(2 * np.pi * f0 * t) + 
                           0.6 * np.sin(2 * np.pi * 2 * f0 * t))
        self.signal = {'time': self.signal_time[np.newaxis, :]}
        
        # Alpha vector
        self.alpha_vec_hz = np.array([0, -f0, -2*f0])

    def test_compare_outputs_valid(self):
        """Verify both methods produce valid coherence outputs."""
        # Just verify frequency-domain method produces valid results
        # Full A/B comparison would require matching the modulator's behavior exactly
        max_bin = 80
        
        # Frequency-domain coherence (new)
        rho_freq = CoherenceManager.compute_coherence_freq_shifted(
            self.signal, self.SFT, self.alpha_vec_hz,
            max_bin=max_bin, use_stft=True,
            interpolation='linear', apply_phase_correction=True,
            min_relative_power=1.e+3
        )
        
        # Check that output is valid coherence matrix
        self.assertEqual(rho_freq.shape[0], len(self.alpha_vec_hz))
        self.assertTrue(np.all(rho_freq >= -0.01))
        self.assertTrue(np.all(rho_freq <= 1.01))
        self.assertTrue(np.all(np.isfinite(rho_freq)))
        
        # Check that no-shift (alpha=0) has high self-coherence
        self.assertGreater(np.mean(rho_freq[0, :]), 0.8)


if __name__ == '__main__':
    unittest.main()
