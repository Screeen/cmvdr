import numpy as np
import warnings
import cmvdr.util.utils as u


class MetricsManager:
    def __init__(self):
        pass

    @staticmethod
    def to_db(x, min_val=1e-15):
        """ Convert a value to decibels (dB). """
        if np.any(x < 0):
            raise ValueError("Input to to_db should be non-negative.")
        return (10. * np.log10(x + min_val) + 300) - 300

    @staticmethod
    def sisdr(reference, estimation):
        """
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

        Args:
            reference: numpy.ndarray, [..., T]
            estimation: numpy.ndarray, [..., T]

        Returns:
            SI-SDR

        [1] SDR– Half-Baked or Well Done?
        http://www.merl.com/publications/docs/TR2019-013.pdf

        # >>> np.random.seed(0)
        # >>> reference = np.random.randn(100)
        # >>> si_sdr(reference, reference)
        # inf
        # >>> si_sdr(reference, reference * 2)
        # inf
        # >>> si_sdr(reference, np.flip(reference))
        # -25.127672346460717
        # >>> si_sdr(reference, reference + np.flip(reference))
        # 0.481070445785553
        # >>> si_sdr(reference, reference + 0.5)
        # 6.3704606032577304
        # >>> si_sdr(reference, reference * 2 + 1)
        # 6.3704606032577304
        # >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
        # nan
        # >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
        # array([6.3704606, 6.3704606])

        """
        estimation, reference = np.broadcast_arrays(estimation, reference)
        reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

        # This is $\alpha$ after Equation (3) in [1].
        optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

        # This is $e_{\text{target}}$ in Equation (4) in [1].
        projection = optimal_scaling * reference

        # This is $e_{\text{res}}$ in Equation (4) in [1].
        noise = estimation - projection

        ratio = np.sum(projection ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1) + 1.e-8)
        return MetricsManager.to_db(ratio)

    @staticmethod
    def compute_stoi(ref_signal, denoised_signal, sr):
        import pystoi
        stoi_value = pystoi.stoi(ref_signal, denoised_signal, sr, extended=True)
        return stoi_value

    @staticmethod
    def compute_pesq(reference, signal, fs):
        import pesq as pypesq  # pip install https://github.com/ludlows/python-pesq/archive/master.zip
        pesq_res = 0
        try:
            if np.sum(np.abs(signal)) > 0:
                if fs == 16000:
                    pesq_res = pypesq.pesq(ref=reference, deg=signal, fs=fs)
                elif fs == 8000:
                    pesq_res = pypesq.pesq(ref=reference, deg=signal, fs=fs, mode='nb')
        except pypesq.NoUtterancesError as e:
            warnings.warn(f"Error computing PESQ: {e}")

        return pesq_res

    @staticmethod
    def compute_dnsmos(signal, sr):
        from speechmos import dnsmos
        dnsmos_res = -np.inf
        if np.sum(np.abs(signal)) > 0:
            try:
                signal = u.normalize_volume(signal, 0.95)
                dnsmos_res = dnsmos.run(signal, sr=sr)
            except Exception as e:
                warnings.warn(f"Error computing DNSMOS: {e}")

        return dnsmos_res
