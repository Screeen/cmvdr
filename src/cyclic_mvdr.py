import numpy as np
import scipy
import warnings
import copy
import src.utils as u
from src.beamformer import Beamformer
from src.covariance_holder import CovarianceHolder

eps = 1e-15


class CyclicMVDR(Beamformer):

    def __init__(self, loadings_cfg, sig_shape_k_m, minimize_noisy_cov_mvdr=True, noise_var_rtf=0):
        super().__init__(loadings_cfg, sig_shape_k_m)
        self.harmonic_info = None

        self.minimize_noisy_cov_mvdr = minimize_noisy_cov_mvdr
        if not minimize_noisy_cov_mvdr:
            warnings.warn('Minimizing the noise covariance matrix. Results will be worse for the cMVDR.')

        # Relative transfer functions (RTFs) for the MVDR beamformer
        self.rtf_est = np.zeros(sig_shape_k_m, dtype=np.complex128, order='F')
        self.rtf_est[:, 0] = 1
        self.noise_var_rtf = noise_var_rtf

    def compute_mvdr_beamformers(self, cov_input_nb, cov_noise_nb, which_variant='blind', speech_rtf_oracle=np.array([])):
        """
        Compute the weights for the MVDR beamformer.
        Parameters
        ----------
        cov_input_nb : (K, M, M) array_like
            Narrowband covariance matrices of the input signal.
        cov_noise_nb : (K, M, M) array_like
            Narrowband covariance matrices of the noise signal.
        which_variant : str
            Variant of the MVDR beamformer to use. Options are 'blind', 'semi-oracle', 'oracle'.
        speech_rtf_oracle : (K, M) array_like, optional
            Oracle relative transfer functions (RTFs) of the speech signal. Required if which_variant is 'oracle' or 'semi-oracle'.
        Returns
        -------
        weights_mvdr : (M, K) ndarray
            MVDR beamforming weights.
        error_flag : (K,) ndarray
            Error flags for each frequency bin.
        cond_num_cov : (K,) ndarray
            Condition numbers of the covariance matrices for each frequency bin.
        singular_values : (K, M) ndarray
            Singular values of the covariance matrices for each frequency bin.
        Notes
        -----
        The MVDR beamformer minimizes the output power while maintaining a distortionless response in the direction
        of the desired signal. The beamforming weights are computed using the covariance matrices of the input
        and noise signals, as well as the relative transfer functions (RTFs) of the desired signal.
        The 'blind' variant estimates the RTFs using the generalized eigenvalue decomposition (GEVD) of the covariance matrices.
        """

        # which_variant: 'blind', 'semi-oracle', 'oracle'
        K_nfft, M = cov_input_nb.shape[:2]
        weights_mvdr = np.zeros((M, K_nfft), dtype=np.complex128)
        error_flag = np.zeros(K_nfft, dtype=bool)
        cond_num_cov = np.zeros(K_nfft)
        singular_values = np.zeros((K_nfft, M))

        if M == 1:  # Single-channel case: no beamforming needed
            weights_mvdr[0, :] = 1
            return weights_mvdr, error_flag, cond_num_cov, singular_values

        loading_mat = np.eye(M)
        loadings = self.get_loading_nb(which_variant, cov_input_nb, *self.loadings_cfg)

        # MVDR beamformer (narrowband)
        for kk in range(K_nfft):
            speech_rtf_oracle_kk = speech_rtf_oracle[kk] if np.any(speech_rtf_oracle) else None
            cov_nb_kk = cov_input_nb[kk] if self.minimize_noisy_cov_mvdr else cov_noise_nb[kk]
            rtf = self.estimate_rtf_or_get_oracle_mvdr(cov_input_nb[kk], cov_noise_nb[kk], speech_rtf_oracle_kk,
                                                       which_variant, kk)

            # cond_num_cov[kk] = np.linalg.cond(cov_nb_kk + loadings[kk] * loading_mat)
            # singular_values[kk] = np.linalg.svd(cov_nb_kk + loadings[kk] * loading_mat, compute_uv=False)

            # (cov_nb_kk)^-1 * atf_mod_kk
            cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_nb_kk + loadings[kk] * loading_mat, rtf, assume_a='pos')
            weights_mvdr[:, kk] = np.squeeze(cov_nb_kk_inv_rtf / (np.conj(rtf).T @ cov_nb_kk_inv_rtf))

        return weights_mvdr, error_flag, cond_num_cov, singular_values

    def compute_cyclic_mvdr_beamformers(self, cov_dict, which_variant, processed_bins, speech_rtf_oracle=np.array([]),
                                        use_pseudo_cov=False, name_input_sig='noisy'):
        """ Compute the weights for the cyclic MVDR beamformer (cMVDR). """

        cov_input_wb = cov_dict[name_input_sig+'_wb']
        cov_input_nb = cov_dict[name_input_sig+'_nb']
        cov_noise_wb = cov_dict['noise_wb']
        cov_noise_nb = cov_dict['noise_nb']

        K_nfft, M = cov_input_nb.shape[:2]
        P_max = cov_input_wb.shape[-1] // M
        P_all = self.harmonic_info.get_num_shifts_all_frequencies()
        loadings = self.get_loading_wb(cov_input_wb, *self.loadings_cfg, P_all, M=M)
        pseudo_cov_factor = 2 if use_pseudo_cov else 1

        eye_wb = np.eye(pseudo_cov_factor * M * P_max)
        eye_nb = np.eye(M)

        weights = np.zeros((pseudo_cov_factor * M * P_max, K_nfft), dtype=np.complex128, order='F')
        error_flag = np.zeros(K_nfft, dtype=bool)

        loadings_nb = self.get_loading_nb(which_variant, cov_input_nb, *self.loadings_cfg)

        cond_num_cov = np.zeros(K_nfft)
        singular_values = np.zeros((K_nfft, M * P_max))

        # cMVDR beamformer
        for kk in range(K_nfft):
            P = P_all[kk] if P_all.size > 0 else 1
            speech_rtf_oracle_kk = speech_rtf_oracle[kk] if np.any(speech_rtf_oracle) else None
            rtf = self.estimate_rtf_or_get_oracle_mvdr(cov_input_nb[kk], cov_noise_nb[kk], speech_rtf_oracle_kk,
                                                       which_variant, kk)
            if kk not in processed_bins:  # Fall back to MVDR for the non-harmonic bins
                cov_kk = cov_input_nb[kk] if self.minimize_noisy_cov_mvdr else cov_noise_nb[kk]
                cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_kk + loadings_nb[kk] * eye_nb, rtf, assume_a='pos')
                weights[:M, kk] = np.squeeze(cov_nb_kk_inv_rtf / (np.conj(rtf).T @ cov_nb_kk_inv_rtf).real)

            else:  # For the harmonic bins, use the cMVDR

                if not use_pseudo_cov:  # use normal covariance matrix E{xx^H}
                    cov_kk = cov_input_wb[kk] if self.minimize_noisy_cov_mvdr else cov_noise_wb[kk]
                    rtf_padded = np.concatenate((rtf, np.zeros(M * P - M)))
                    try:
                        cov_wb_kk_inv_rtf = scipy.linalg.solve(
                            cov_kk[:M * P, :M * P] + loadings[kk] * eye_wb[:M * P, :M * P],
                            rtf_padded, assume_a='pos')

                    except np.linalg.LinAlgError:
                        warnings.warn(f"LinAlgError for bin {kk} in cMVDR. Using narrowband weights.")
                        P = 1
                        rtf_padded = rtf_padded[:M * P]
                        cov_wb_kk_inv_rtf = scipy.linalg.solve(
                            cov_kk[:M * P, :M * P] + loadings[kk] * eye_wb[:M * P, :M * P],
                            rtf_padded[:M * P], assume_a='pos')
                    weights[:M * P, kk] = cov_wb_kk_inv_rtf / (np.conj(rtf_padded).T @ cov_wb_kk_inv_rtf).real

                else:  # use augmented covariance matrix with x__ = [x x^*]
                    A = cov_input_wb[kk, :M * P, :M * P] + loadings[kk] * eye_wb[:M*P, :M*P]
                    B = cov_dict[name_input_sig + '_pseudo'][kk, :M * P, :M * P]
                    two_MP = 2 * M * P
                    n_zeros = M * (P - 1)
                    rtf_padded_half = np.concatenate((rtf, np.zeros(n_zeros)))
                    rtf_padded = np.concatenate((rtf, np.zeros(n_zeros), np.conj(rtf), np.zeros(n_zeros)))

                    # With Schur complement
                    cov_wb_kk_inv_rtf = self.solve_via_schur(A, B,
                                                                rtf_padded_half, np.conj(rtf_padded_half))
                    weights[:two_MP, kk] = cov_wb_kk_inv_rtf / (np.conj(rtf_padded).T @ cov_wb_kk_inv_rtf).real

                    # Brute force option (TODO: test if identical to solve_via_schur)
                    # cov_aug = np.block([ [A, B], [B.conj(), A.conj()] ])
                    # cov_wb_kk_inv_rtf = scipy.linalg.solve(cov_aug[:two_MP, :two_MP], rtf_padded, assume_a='pos')
                    # weights[:two_MP, kk] = cov_wb_kk_inv_rtf / (np.conj(rtf_padded).T @ cov_wb_kk_inv_rtf).real

            # cond_num_cov[kk] = np.linalg.cond(cov_kk[:M * P, :M * P] + loadings[kk] * eye_wb[:M * P, :M * P])
            # singular_values[kk, :M * P] = np.linalg.svd(cov_kk[:M * P, :M * P] + loadings[kk] * eye_wb[:M * P, :M * P],
            #                                             compute_uv=False)

        return weights, error_flag, cond_num_cov, singular_values

    @staticmethod
    def solve_via_schur(A, B, z1, z2):
        """
        Solve [A  B; B*  A*] [w1; w2] = [z1; z2] using Schur complement
        and Cholesky factorizations for numerical stability.

        Parameters
        ----------
        A : (n, n) array_like
            Hermitian positive definite matrix.
        B : (n, n) array_like
            Symmetric matrix.
        z1 : (n,) array_like
            First block of RHS vector.
        z2 : (n,) array_like
            Second block of RHS vector.

        Returns
        -------
        w : (2n,) ndarray
            Solution vector [w1; w2].
        """

        cho_factor = scipy.linalg.cho_factor
        cho_solve = scipy.linalg.cho_solve

        # Prepare B^H (B=B^T --> B^* = B^H)
        B_conj = np.conj(B)

        # Factor A
        cA, lowA = cho_factor(A, overwrite_a=False, check_finite=True)

        # Compute A^{-1} @ B and A^{-1} @ z1 without forming A^{-1}
        A_inv_B = cho_solve((cA, lowA), B)
        A_inv_z1 = cho_solve((cA, lowA), z1)

        # Form the Schur complement S = A* - B* A^{-1} B.
        # Notice that S = S^H
        S = np.conj(A) - B_conj @ A_inv_B

        # Solve for w2
        rhs2 = z2 - B_conj @ A_inv_z1

        try:
            # Factor S
            cS, lowS = cho_factor(S, overwrite_a=False, check_finite=True)
            w2 = cho_solve((cS, lowS), rhs2)
        except np.linalg.LinAlgError:
            # S_pinv = scipy.linalg.pinvh(S)
            # w2 = S_pinv @ rhs2
            w1 = scipy.linalg.solve(A, z1, assume_a='pos')
            return np.concatenate([w1, np.conj(w1)])

        # Back-substitute to get w1
        w1 = cho_solve((cA, lowA), z1 - B @ w2)

        return np.concatenate([w1, w2])

    def estimate_rtf_or_get_oracle_mvdr(self, cov_noisy_nb, cov_noise_nb, oracle_rtf=None, which_variant='blind',
                                        kk=None):
        # If the RTF is fixed, we can update it every few iterations, not at every iteration.

        if which_variant == 'oracle' or which_variant == 'semi-oracle':  # Oracle relative transfer functions (RTFs)
            # Add noise to estimated rtf
            if self.noise_var_rtf > 0:
                temp = copy.deepcopy(oracle_rtf)
                temp[1:] = oracle_rtf[1:] + np.sqrt(self.noise_var_rtf) * u.circular_gaussian(oracle_rtf[1:].shape)
                return temp
            return oracle_rtf

        elif which_variant == 'blind':  # Estimate RTF using GEVD (covariance whitening)
            if not self.rtf_needs_estimation[kk]:
                return self.rtf_est[kk]

            # Skip RTF estimation if noisy power smaller than noise power (probably a noise-only segment)
            if cov_noisy_nb.shape[0] == 1 or (np.trace(cov_noisy_nb) < np.trace(cov_noise_nb)):
                # rtf = np.zeros_like(self.rtf_est[kk])
                # rtf[0] = 1
                # self.rtf_est[kk] = np.squeeze(rtf)
                self.rtf_needs_estimation[kk] = False
                return self.rtf_est[kk]

            ea, ev = scipy.linalg.eigh(cov_noisy_nb, cov_noise_nb)
            ev_left_signal = cov_noise_nb @ ev[:, -1:]
            if np.any(np.isclose(ev_left_signal, 0)):
                warnings.warn(f"RTF is close to 0 for bin {kk}.")
            rtf = ev_left_signal / ev_left_signal[0]
            self.rtf_est[kk] = np.squeeze(rtf)
            self.rtf_needs_estimation[kk] = False
            return self.rtf_est[kk]
        else:
            raise ValueError(f"Unknown variant for MVDR beamformer: {which_variant}")

    def check_if_rtf_needs_estimation(self, idx_chunk=0, warmup_chunks=Beamformer.rtf_est_warmup_chunks,
                                      interval_chunks=Beamformer.rtf_est_interval_chunks):
        """ Check if RTF needs to be estimated for the current chunk."""
        if self.rtf_est.shape[1] == 1:  # Single-channel case: no RTF estimation needed
            return np.zeros_like(self.rtf_needs_estimation)
        if idx_chunk < warmup_chunks or (interval_chunks > 0 and idx_chunk % interval_chunks == 0):
            return np.ones_like(self.rtf_needs_estimation)
        return np.zeros_like(self.rtf_needs_estimation)
