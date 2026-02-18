"""
Tests for frequency bin mapping between high-res coherence and low-res beamforming.
"""

import unittest
import numpy as np
from cmvdr.estimation.coherence_manager import CoherenceManager


class TestFrequencyBinMapping(unittest.TestCase):
    """Test that high-res coherence bins are correctly mapped to low-res beamforming bins."""
    
    def test_high_res_to_low_res_mapping(self):
        """Test mapping from high-resolution coherence to low-resolution beamforming."""
        # Setup: High-res coherence (2048 FFT) vs low-res beamforming (512 FFT)
        fs = 16000
        nfft_coherence = 2048
        nfft_beamforming = 512
        
        # Frequency resolutions
        delta_f_coherence = fs / nfft_coherence  # 7.8125 Hz
        delta_f_beamforming = fs / nfft_beamforming  # 31.25 Hz
        nfft_real_coherence = nfft_coherence // 2 + 1  # 1025
        nfft_real_beamforming = nfft_beamforming // 2 + 1  # 257
        
        # Create simple coherence matrix with high coherence at a few high-res bins
        alpha_vec_hz = np.array([0, -100, -200])  # 3 modulation frequencies
        rho = np.zeros((len(alpha_vec_hz), nfft_real_coherence))
        
        # Set high coherence at specific high-res bins
        high_res_bins = [0, 64, 128, 256, 512]  # Mix of low and high frequency bins
        for bin_idx in high_res_bins:
            rho[:, bin_idx] = 1.0  # All alphas have high coherence at these bins
        
        thr = 0.5
        P_max_cfg = 3
        
        # Call with frequency mapping
        harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
            alpha_vec_hz, rho, thr, P_max_cfg, 
            nfft_real=nfft_real_beamforming,
            delta_f_coherence=delta_f_coherence,
            delta_f_beamforming=delta_f_beamforming
        )
        
        # Verify that harmonic bins are within beamforming bounds
        self.assertTrue(np.all(harm_info.harmonic_bins < nfft_real_beamforming),
                       f"Harmonic bins {harm_info.harmonic_bins} exceed beamforming resolution {nfft_real_beamforming}")
        
        # Verify mapping: high-res bin -> frequency -> low-res bin
        expected_low_res_bins = []
        for high_res_bin in high_res_bins:
            freq_hz = high_res_bin * delta_f_coherence
            low_res_bin = int(np.round(freq_hz / delta_f_beamforming))
            if low_res_bin < nfft_real_beamforming:
                expected_low_res_bins.append(low_res_bin)
        
        expected_low_res_bins = np.array(expected_low_res_bins)
        
        # Check that harmonic bins match expected mapping
        np.testing.assert_array_equal(
            np.sort(harm_info.harmonic_bins),
            np.sort(expected_low_res_bins),
            err_msg="Harmonic bins don't match expected frequency mapping"
        )
    
    def test_no_mapping_when_resolutions_match(self):
        """Test that no mapping occurs when coherence and beamforming use same resolution."""
        # Both use same resolution
        fs = 16000
        nfft = 512
        delta_f = fs / nfft
        nfft_real = nfft // 2 + 1  # 257
        
        alpha_vec_hz = np.array([0, -100, -200])
        rho = np.zeros((len(alpha_vec_hz), nfft_real))
        
        # Set high coherence at specific bins
        coherent_bins = [0, 10, 20, 50, 100]
        for bin_idx in coherent_bins:
            rho[:, bin_idx] = 1.0
        
        thr = 0.5
        P_max_cfg = 3
        
        # Call with same delta_f (or None, which means same)
        harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
            alpha_vec_hz, rho, thr, P_max_cfg, nfft_real,
            delta_f_coherence=delta_f,
            delta_f_beamforming=delta_f
        )
        
        # Verify bins are unchanged
        self.assertEqual(len(harm_info.harmonic_bins), len(coherent_bins))
        np.testing.assert_array_equal(
            np.sort(harm_info.harmonic_bins),
            np.array(coherent_bins),
            err_msg="Bins should be unchanged when resolutions match"
        )
    
    def test_default_no_mapping(self):
        """Test backward compatibility: no mapping when delta_f parameters not provided."""
        # Default behavior (no delta_f parameters)
        nfft_real = 100
        alpha_vec_hz = np.array([0, -100, -200])
        rho = np.zeros((len(alpha_vec_hz), nfft_real))
        
        # Set high coherence at specific bins
        coherent_bins = [5, 15, 25, 50, 75]
        for bin_idx in coherent_bins:
            rho[:, bin_idx] = 1.0
        
        thr = 0.5
        P_max_cfg = 3
        
        # Call without delta_f parameters (backward compatible)
        harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
            alpha_vec_hz, rho, thr, P_max_cfg, nfft_real
        )
        
        # Verify bins are as provided (no mapping)
        self.assertEqual(len(harm_info.harmonic_bins), len(coherent_bins))
        np.testing.assert_array_equal(
            np.sort(harm_info.harmonic_bins),
            np.array(coherent_bins),
            err_msg="Bins should be unchanged when delta_f not provided"
        )
    
    def test_skip_bins_beyond_beamforming_resolution(self):
        """Test that high-frequency coherence bins beyond beamforming range are skipped."""
        # High-res coherence with bins that exceed beamforming range
        fs = 16000
        nfft_coherence = 2048
        nfft_beamforming = 512
        
        delta_f_coherence = fs / nfft_coherence
        delta_f_beamforming = fs / nfft_beamforming
        nfft_real_coherence = nfft_coherence // 2 + 1  # 1025
        nfft_real_beamforming = nfft_beamforming // 2 + 1  # 257
        
        alpha_vec_hz = np.array([0, -100])
        rho = np.zeros((len(alpha_vec_hz), nfft_real_coherence))
        
        # Set high coherence at bins that would map beyond beamforming range
        # High-res bin 1000 -> freq = 1000 * 7.8125 = 7812.5 Hz
        # Low-res bin would be round(7812.5 / 31.25) = 250
        # But max low-res bin is 256, so bins close to limit should work
        high_res_bins = [0, 100, 500, 900, 1000]  # Last two might be beyond range
        for bin_idx in high_res_bins:
            if bin_idx < nfft_real_coherence:
                rho[:, bin_idx] = 1.0
        
        thr = 0.5
        P_max_cfg = 2
        
        harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
            alpha_vec_hz, rho, thr, P_max_cfg,
            nfft_real=nfft_real_beamforming,
            delta_f_coherence=delta_f_coherence,
            delta_f_beamforming=delta_f_beamforming
        )
        
        # All harmonic bins should be within beamforming bounds
        self.assertTrue(np.all(harm_info.harmonic_bins < nfft_real_beamforming),
                       f"Some harmonic bins exceed beamforming resolution")
        self.assertTrue(np.all(harm_info.harmonic_bins >= 0),
                       f"Some harmonic bins are negative")


if __name__ == '__main__':
    unittest.main()
