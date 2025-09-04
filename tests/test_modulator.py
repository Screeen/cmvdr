# We test the modulator class by modulating a signal and then demodulating it.
# The modulated signal should be the same as the original signal.
#
# The modulator class uses the spectral correlation estimator to modulate and demodulate the signal.
import copy
import unittest
import numpy as np
import scipy

from cmvdr.estimation.modulator import Modulator
from cmvdr.util import globs as gs, utils as u

gs.rng, _ = gs.compute_rng(seed_is_random=True)


def assert_allclose(x, y, rtol=1e-5, atol=1e-8):
    return np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def assert_isreal(x, rtol=1e-5, atol=1e-8):
    return np.testing.assert_allclose(x.flatten().real, x.flatten(), rtol=rtol, atol=atol)


class TestModulator(unittest.TestCase):

    def get_impulse_response(self, h_len):
        # Create a random impulse response
        h_len_nz = h_len // 2
        h = np.random.randn(h_len_nz)
        decaying_window = np.exp(-np.linspace(0, 1, h_len_nz))
        h = h * decaying_window
        h = u.pad_last_dim(h, h_len - h_len // 4)
        h = u.pad_last_dim(h, h_len, prepad=True)
        return h

    def get_random_sin(self, dry_len, fs, post_pad_amount=0):
        # Create a random signal
        dry_len_nz = dry_len - post_pad_amount
        s = 1e-6 * np.random.randn(1, dry_len_nz) \
            + 0.01 * np.cos(2 * np.pi * 5 * np.arange(dry_len_nz) / fs) \
            + 0.1 * np.cos(2 * np.pi * 33 * np.arange(dry_len_nz) / fs)
        s = u.pad_last_dim(s, dry_len, prepad=False)
        return s

    def test_modulator(self):
        show_plots = False
        fs = 4000.
        max_len_samples = int(fs * 0.1)
        alpha_vec_hz = np.array([0, -10, 500])
        mod_idx = 1
        modulator = Modulator(max_len_samples, fs, [alpha_vec_hz])
        x = self.get_random_sin(max_len_samples, fs)
        X = np.fft.fft(x)
        X = np.fft.fftshift(X, axes=-1)

        x_mod = modulator.modulate(x[np.newaxis])
        X_mod = np.fft.fftshift(np.fft.fft(x_mod), axes=-1)

        x_demod = modulator.demodulate(x_mod)
        X_demod = np.fft.fft(x_demod)
        X_demod = np.fft.fftshift(X_demod, axes=-1)

        if show_plots:
            u.plot([np.log10(np.abs(X[0])), np.log10(np.abs(X_mod[0])), np.log10(np.abs(X_demod[0, 0])),
                    np.log10(np.abs(X_demod[0, 1]))],
                   titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
                   subplot_height=1.5, time_axis=False)

            # Same as above, but real part
            # u.plot([X[0].real, X_mod[0].real, X_demod[0,0].real, X_demod[0,1].real],
            #        titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
            #        subplot_height=1.5, time_axis=False)

            # Same as above, but imaginary part
            # u.plot([X[0].imag, X_mod[0].imag, X_demod[0,0].imag, X_demod[0,1].imag],
            #        titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
            #        subplot_height=1.5, time_axis=False)

            # u.plot([np.unwrap(np.angle(X[0])), np.unwrap(np.angle(X_mod[0])), np.unwrap(np.angle(X_demod[0, 0])),
            #         np.unwrap(np.angle(X_demod[0, 1]))],
            #        titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
            #        subplot_height=1.5, time_axis=False)

        # u.plot([x[0], x_demod[0, 0], x_demod[0, 1]], titles=['x', 'x_demod0', 'x_demod1'], subplot_height=1.5,
        #        time_axis=True)

        assert_allclose(np.angle(X[0]), np.angle(X_demod[0, mod_idx]), atol=1e-12)
        assert_allclose(np.abs(X[0]), np.abs(X_demod[0, mod_idx]), atol=1e-12)
        assert_allclose(x.flatten(), x_demod[0, mod_idx].flatten(), atol=1e-12)

    def test_convolution(self):
        # Test 1:
        # sh_mod_1 = modulate( s conv h )
        # sh_mod_2 = modulate( s ) conv modulate( h )
        # sh_mod_1 == sh_mod_2?
        # Notice that this works with 'full' convolution but not with 'same' or 'valid' convolution.

        # Test 2:
        # sh_demod_1 = demodulate( sh_mod_1 )
        # sh_demod_2 = demodulate( sh_mod_2 )
        # sh_demod_1 == sh_demod_2 == s * h?

        mic0 = 0
        convolution_mode = 'full'
        fs = 4000.
        dry_len = int(fs * 0.1)
        h_len = int(fs * 0.02)
        wet_len = dry_len + h_len - 1

        alpha_vec_hz = np.atleast_1d(np.array([0, 500, -50]))
        P = len(alpha_vec_hz)
        modulator = Modulator(wet_len, fs, [alpha_vec_hz])

        s = self.get_random_sin(dry_len, fs)
        h = self.get_impulse_response(h_len)

        # Convolve the signal with the impulse response
        sh = scipy.signal.convolve(s[mic0], h, mode=convolution_mode)
        sh_mod_1 = modulator.modulate(sh[np.newaxis, np.newaxis])

        # Modulate the signal and impulse response
        s_mod = modulator.modulate(s[np.newaxis])
        h_mod = modulator.modulate(h[np.newaxis, np.newaxis])

        sh_mod_2 = np.zeros((1, P, wet_len), dtype=complex)
        for pp in range(P):
            sh_mod_2[mic0, pp, :] = scipy.signal.convolve(s_mod[mic0, pp, :], h_mod[mic0, pp, :], mode=convolution_mode)

        assert_allclose(sh_mod_1, sh_mod_2)

        # Demodulate the convolved signal
        sh_demod_1 = modulator.demodulate(sh_mod_1)
        sh_demod_2 = modulator.demodulate(sh_mod_2)

        assert_isreal(sh_demod_1)
        assert_isreal(sh_demod_2)

        for pp in range(P):
            assert_allclose(sh, sh_demod_1[mic0, pp])
            assert_allclose(sh, sh_demod_2[mic0, pp])

    def test_non_modulated_transfer_function(self):
        # Test 1: without modulations.
        # Take a signal, convolve it with a (short) impulse response. Alternatively, take the signal and the impulse
        # response, transform them to frequency domain, multiply them, and transform back to time domain.
        # The results should be the same.

        mic0 = 0
        convolution_mode = 'full'
        fs = 4000.
        fft_size = 1024
        dry_len = fft_size
        h_len = 128

        s = self.get_random_sin(dry_len, fs, post_pad_amount=h_len - 1)
        h = self.get_impulse_response(h_len)
        sh1 = scipy.signal.convolve(s[mic0], h, mode=convolution_mode)
        sh1 = sh1[..., :fft_size]

        S = np.fft.fft(s[mic0], n=fft_size)
        H = np.fft.fft(h, n=fft_size)
        SH = S * H
        sh2 = np.fft.ifft(SH).real
        # sh2 = np.roll(sh2, 2*h_len + 1)

        off = h_len
        if off >= fft_size:
            raise ValueError(f"{off=} should be less than {fft_size=}. Increase fft_size or decrease h_len.")

        sh1_c = sh1[off:-off]
        sh2_c = sh2[off:-off]

        # tolerance needs to be high as (time-convolution <-> freq-domain multiplication) is not exact
        assert_allclose(sh1_c, sh2_c, atol=1e-4)
        # u.plot([sh1, sh2], titles=['sh1', 'sh2'], subplot_height=1.5, time_axis=False)
        # u.plot([sh1_c, sh2_c], titles=['sh1_c', 'sh2_c'], subplot_height=1.5, time_axis=False)

    def test_modulated_transfer_function(self):

        # Test 2: with modulations.
        # Take a signal, modulate it. Take a impulse response, modulate it.
        #
        # First approach: Convolve the modulated signal with the modulated impulse response. Transform the convolved
        # signal to the frequency domain.
        # Second approach: transform the modulated signal and the modulated impulse response to the frequency domain,
        # multiply them. The results should be the same.
        mic0 = 0
        convolution_mode = 'full'
        fs = 4000.
        fft_size = 512
        dry_len = fft_size
        h_len = 200
        wet_len = dry_len + h_len - 1

        alpha_vec_hz = np.array([0, 96, 320, -96])
        modulator = Modulator(wet_len, fs, [alpha_vec_hz])
        P = len(modulator.alpha_vec_hz_)

        s = self.get_random_sin(dry_len, fs, post_pad_amount=h_len - 1)
        h = self.get_impulse_response(h_len)

        # First approach: Convolve the modulated signal with the modulated impulse response.
        # Transform the convolved signal to the frequency domain.
        s_mod = modulator.modulate(s[np.newaxis])
        h_mod = modulator.modulate(h[np.newaxis, np.newaxis])

        sh_mod1 = np.zeros((1, P, wet_len), complex)
        for pp in range(P):
            sh_mod1[mic0, pp, :] = scipy.signal.convolve(s_mod[mic0, pp, :], h_mod[mic0, pp, :], mode=convolution_mode)

        sh_demod_1 = modulator.demodulate(sh_mod1)
        assert_isreal(sh_demod_1)
        sh_demod_1 = sh_demod_1.real

        # Second approach: transform the modulated signal and the modulated impulse response to the frequency domain,
        # multiply them.
        S_mod = np.fft.fft(s_mod, n=fft_size)
        H_mod = np.fft.fft(h_mod, n=fft_size)
        SH_mod2 = S_mod * H_mod
        sh_mod2 = np.fft.ifft(SH_mod2)
        sh_demod_2 = modulator.demodulate(sh_mod2)

        #
        assert_isreal(sh_demod_2)
        sh_demod_2 = sh_demod_2.real
        sh_demod_2 = u.pad_last_dim(sh_demod_2[0], wet_len, prepad=False)[np.newaxis]

        assert_allclose(sh_demod_1.flatten(), sh_demod_2.flatten(), atol=1e-4)

        sh1 = scipy.signal.convolve(s[mic0], h, mode=convolution_mode)
        sh1 = u.pad_last_dim(sh1, wet_len)
        for pp in range(P):
            assert_allclose(sh1, sh_demod_1[mic0, pp])
            assert_allclose(sh1, sh_demod_2[mic0, pp])

        # The signals look similar, but they are different nearby the edges, unless we pad the dry signal with zeros before convolution.
        # for pp in range(P):
        #     sh_demod_1_c = sh_demod_1[mic0, pp]
        #     sh_demod_2_c = sh_demod_2[mic0, pp]
        #     sh1_c = sh1
        #     u.plot([sh1_c, sh_demod_1_c, sh_demod_2_c],
        #            titles=[f'sh1_c, {alpha_vec_hz[pp]} Hz', f'sh_demod_1_c, {alpha_vec_hz[pp]} Hz', f'sh_demod_2_c, {alpha_vec_hz[pp]} Hz'],
        #            subplot_height=1.5, time_axis=False)

    def test_inharmonic_modulator(self):
        """

        # New implementation of modulator
        mod2 = modulator.Modulator(max_len, fs, alpha_mods_list1, use_filters=use_filters,
                                   fast_version=True)
        signals_unproc_copy = mod2.compute_reshaped_modulated_signals(signals_unproc_copy, SFT,
                                                                      signals_to_modulate,
                                                                      harmonic_info.num_shifts_per_set)

        # Verify that the two implementations give same result
        for key, val in signals_unproc1.items():
            if 'mod_stft_3d' in val:
                if not np.allclose(val['mod_stft_3d'], signals_unproc_copy[key]['mod_stft_3d']):
                    ii = np.argwhere(
                        np.abs(val['mod_stft_3d'] - signals_unproc_copy[key]['mod_stft_3d']) > 1.e-3)
                    print(ii)
                assert np.allclose(val['mod_stft_3d'], signals_unproc_copy[key]['mod_stft_3d'])
                assert np.allclose(val['mod_stft_3d_conj'], signals_unproc_copy[key]['mod_stft_3d_conj'])
        :return:
        """

        fs = 8000.
        fft_size = int(gs.rng.choice([256, 512, 1024, 2048]))
        fft_size_real = fft_size // 2 + 1
        max_len = int(fs * 1.)
        M = gs.rng.integers(1, 10)
        P_max = gs.rng.integers(1, 4)
        SFT = scipy.signal.ShortTimeFFT(win=scipy.signal.windows.hamming(fft_size), hop=fft_size // 2,
                                        fs=fs, fft_mode='twosided')

        signal_has_harmonic_list = [False, True]

        for signal_has_harmonic in signal_has_harmonic_list:
            if signal_has_harmonic:
                alpha_mods_list1 = [np.r_[0, 96, 320, -96],
                                    np.r_[0, 96.00001, gs.rng.integers(90, 100), -96],
                                    np.r_[0, 3, -96.00001, gs.rng.integers(90, 100), -99]]
                num_shifts_per_set = np.array([P_max, max(1, P_max-1), 1])
            else:
                num_shifts_per_set = np.array([1])
                alpha_mods_list1 = [np.array([0])]

            signals_to_modulate = ['noisy']

            # sig_dict1['noisy']['stft'].shape[0]
            sig_dict1 = {'noisy': {'time': gs.rng.random((M, max_len // 3))}}
            sig_dict1['noisy']['stft'] = SFT.stft(x=sig_dict1['noisy']['time'])[:, :fft_size_real]
            sig_dict1['noisy']['stft_conj'] = np.conj(sig_dict1['noisy']['stft'])
            sig_dict2 = copy.deepcopy(sig_dict1)

            # With this inputs are different and it will fail
            # sig_dict2['noisy']['time'] = sig_dict2['noisy']['time'] + 0.02

            # Old implementation of modulator
            mod1 = Modulator(max_len, fs, alpha_mods_list1)
            sig_dict1 = mod1.compute_modulated_signals(sig_dict1, SFT, P_max=P_max,
                                                             names_signals_to_modulate=signals_to_modulate)
            sig_dict1 = mod1.rearrange_modulated_signals(sig_dict1, signals_to_modulate, num_shifts_per_set, P_max)

            # New implementation
            alpha_mods_list2 = copy.deepcopy(alpha_mods_list1)

            # With this change modulations are different and it will fail
            # alpha_mods_list2[0][0] = 3

            mod2 = Modulator(max_len, fs, alpha_mods_list2, fast_version=True)
            np.testing.assert_allclose(sig_dict1['noisy']['time'], sig_dict2['noisy']['time'],
                                       err_msg="Input should be equal for meaningful comparison")
            sig_dict2 = mod2.compute_reshaped_modulated_signals(sig_dict2, SFT, P_max, signals_to_modulate,
                                                                num_shifts_per_set)

            for d1, d2 in zip(sig_dict1.values(), sig_dict2.values()):
                np.testing.assert_allclose(d1['mod_stft_3d'], d2['mod_stft_3d'])
                np.testing.assert_allclose(d1['mod_stft_3d_conj'], d2['mod_stft_3d_conj'])
