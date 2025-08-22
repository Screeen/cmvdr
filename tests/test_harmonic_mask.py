import copy
import unittest

import numpy as np

from cmvdr.coherence_manager import CoherenceManager
from cmvdr import globs as gs, harmonic_info, covariance_holder

gs.rng, _ = gs.compute_rng(seed_is_random=True)


class TestModulator(unittest.TestCase):

    @staticmethod
    def generate_covariance_equicorrelated(cov_shape: tuple, corr_coefficient: float = 0, variances: np.ndarray = 1):
        """
        Generate a covariance matrix with given correlation coefficient and variance.
        All cross-elements are found as np.sqrt(r[i, i] * r[j, j]), that is an upper bound on the cross-correlation,
        and then multiplied by corr_coefficient. Diagonal elements are preserved.
        :param variances: the variance (diagonal elements of the covariance matrix)
        :param corr_coefficient: the correlation coefficient (between -1 and 1), determines off-diagonal elements
        :param cov_shape: the shape of the covariance matrix
        :return: generated covariance matrix
        """

        if cov_shape[0] != cov_shape[1]:
            raise ValueError(f'cov_shape must be a tuple of two equal integers, but is {cov_shape}')

        if len(cov_shape) != 2:
            raise ValueError(f'cov_shape must be a tuple of two integers, but is {cov_shape}')

        if np.isscalar(variances):
            variances = np.ones(cov_shape[0]) * variances

        if len(variances) != cov_shape[0]:
            raise ValueError(f'variances must have length {cov_shape[0]}')

        r = np.diag(variances).astype(complex)

        if corr_coefficient != 0:
            corr_coefficient_mat = np.ones(cov_shape, dtype=complex) * corr_coefficient
            for ii in range(cov_shape[0]):
                for jj in range(ii + 1, cov_shape[1]):
                    r[ii, jj] = np.sqrt(r[ii, ii] * r[jj, jj]) * corr_coefficient_mat[ii, jj]
                    r[jj, ii] = r[ii, jj].conj()

        return r

    def test_corr_values(self):

        ch = covariance_holder.CovarianceHolder()
        M = 2
        P_max = 4
        K = 10
        harmonic_sets = np.ones(K) * -1
        harmonic_sets[2:4] = 0
        harmonic_sets[5:7] = 1
        harmonic_sets[9] = 2
        num_shifts = np.array([2, 3, 2])
        P = int(np.min(num_shifts))

        harm_info = harmonic_info.HarmonicInfo(harmonic_sets=harmonic_sets)
        harm_info.num_shifts_per_set = num_shifts

        ch.noisy_wb = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)
        for kk in range(K):
            ch.noisy_wb[kk] = self.generate_covariance_equicorrelated((M * P_max, M * P_max,),
                                                                      gs.rng.uniform(0, 1),
                                                                      gs.rng.uniform(1, 10, M * P_max))

        coh_man = CoherenceManager()
        coherence = coh_man.covariance_to_coherence(ch.noisy_wb)
        for rho in [0, 0.5, 1]:
            mask = coh_man.get_correlation_masks(coherence, harm_info, M, rho)
            self.assertTrue(mask.size > 0)
            if rho == 0:
                np.testing.assert_array_equal(mask[:, :M * P], True)
            elif rho == 1:
                self.assertTrue(np.all(mask[:, :M]))
                self.assertFalse(np.all(mask[:, M:]))
            else:
                np.testing.assert_array_equal(mask[:, :M], True)

    def test_apply_changes(self):

        ch = covariance_holder.CovarianceHolder()
        M = 2
        P_max = 4
        K = 10
        harmonic_sets = np.ones(K, dtype=int) * -1
        harmonic_sets[2:4] = 0
        harmonic_sets[5:7] = 1
        harmonic_sets[9] = 2
        signals_unproc = {'noisy': {'stft': np.zeros((M, K))}}
        alpha_mods_sets = [np.array([0, 1]), np.array([0, 3, -5]), np.array([0, 9])]

        harm_info = harmonic_info.HarmonicInfo(harmonic_sets=harmonic_sets,
                                               alpha_mods_sets=alpha_mods_sets)
        coh_man = CoherenceManager()

        ch.noisy_wb = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)
        ch.noisy_nb = np.zeros((K, M, M), dtype=np.complex128)
        for kk in range(K):
            ch.noisy_wb[kk] = self.generate_covariance_equicorrelated((M * P_max, M * P_max,),
                                                                      gs.rng.uniform(0, 1),
                                                                      gs.rng.uniform(1, 10, M * P_max))

        coherence = coh_man.covariance_to_coherence(ch.noisy_wb)
        correlation_masks = coh_man.get_correlation_masks(coherence, harm_info, M,
                                                                             threshold_harmonic=gs.rng.uniform(0, 1))

        harm_info_copy = copy.deepcopy(harm_info)
        harm_info, temp_out, signals_unproc = coh_man.apply_local_coherence_masks(correlation_masks,
                                                                                  harm_info,
                                                                                  ch.noisy_wb,
                                                                                  signals_unproc)
        ch.noisy_wb = temp_out[0]

        # Number of harmonic sets must not change
        self.assertEqual(len(harm_info.alpha_mods_sets), len(harm_info_copy.alpha_mods_sets))

        # Size of matrices must not change
        self.assertEqual(ch.noisy_wb.shape, (K, M * P_max, M * P_max))
        self.assertEqual(ch.noisy_nb.shape, (K, M, M))

        for cc in range(len(alpha_mods_sets)):
            # This should be consistent
            self.assertEqual(len(harm_info.alpha_mods_sets[cc]), harm_info.num_shifts_per_set[cc])

            # Number of modulations cannot be larger after applying the masks
            self.assertLessEqual(len(harm_info.alpha_mods_sets[cc]), len(harm_info_copy.alpha_mods_sets[cc]))
