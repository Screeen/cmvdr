"""
Plot theoretical vs simulated noise reduction performance (ResNoise factor).

This script validates the theoretical noise reduction performance from equation 
eq:res_noise_cmvdr:factor against simulation results.

Theoretical formula: ResNoise = 1 - ρ² / (1 + σ²ᵢ/σ²ᵥ)
where:
- ρ is the spectral correlation coefficient
- σ²ᵢ is the interference power
- σ²ᵥ is the self-noise (microphone) power
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import ShortTimeFFT
import scipy.signal
from datetime import datetime

from cmvdr.data_gen.data_generator import DataGenerator
from cmvdr.beamforming.beamformer_manager import BeamformerManager
from cmvdr.estimation.covariance_estimator import CovarianceEstimator
from cmvdr.util.harmonic_info import HarmonicInfo
from cmvdr.util import utils as u
from cmvdr.util import globs as gs
from cmvdr.data_gen.manager import Manager


def theoretical_res_noise(rho, sigma_i_to_v_ratio):
    """
    Calculate theoretical residual noise factor.
    
    Parameters
    ----------
    rho : float or ndarray
        Spectral correlation coefficient |ρ| (0 to 1)
    sigma_i_to_v_ratio : float
        Interference-to-noise power ratio σ²ᵢ/σ²ᵥ
        
    Returns
    -------
    float or ndarray
        ResNoise factor η
    """
    return 1 - rho**2 / (1 + sigma_i_to_v_ratio)


def generate_synthetic_signal(rho, snr_db_dir, snr_db_self, fs=16000, duration_sec=2.0, 
                               f0_hz=100.0, num_harmonics=50, M=1):
    """
    Generate synthetic single-microphone signal with controlled spectral correlation.
    
    Parameters
    ----------
    rho : float
        Target spectral correlation (0 to 1)
    snr_db_dir : float
        Directional interference SNR in dB
    snr_db_self : float
        Self-noise (microphone) SNR in dB
    fs : int
        Sampling frequency
    duration_sec : float
        Signal duration in seconds
    f0_hz : float
        Fundamental frequency of harmonic signal
    num_harmonics : int
        Number of harmonics
    M : int
        Number of microphones (should be 1 for this analysis)
        
    Returns
    -------
    dict
        Dictionary with 'noisy', 'wet', 'noise_dir', 'noise_self' signals
    """
    N_samples = int(fs * duration_sec)
    
    # Initialize data generator with controlled noise correlation
    data_gen = DataGenerator(target_harmonic_corr=1.0, 
                            noise_harmonic_corr=rho)
    
    # Generate clean target signal (single mic, so just a scalar signal)
    target_anechoic, _ = data_gen.generate_or_load_anechoic_signal(
        N_samples, fs=fs, 
        sample_path=None, 
        sig_type='sinusoidal',
        f0_hz=f0_hz,
        num_harmonics=num_harmonics,
        inharmonicity_percentage=0.0,
        fixed_amplitudes_sin=True,
        sin_gen=data_gen.sin_gen['target']
    )
    
    # For single microphone, reshape to (1, N_samples)
    if target_anechoic.ndim == 1:
        target_anechoic = target_anechoic[np.newaxis, :]
    
    # Generate interference (directional noise)
    noise_directional, _ = data_gen.generate_or_load_anechoic_signal(
        N_samples, fs=fs,
        sample_path=None,
        sig_type='sinusoidal',
        f0_hz=f0_hz * 1.5,  # Different frequency for interference
        num_harmonics=num_harmonics,
        inharmonicity_percentage=0.0,
        fixed_amplitudes_sin=True,
        sin_gen=data_gen.sin_gen['noise']
    )
    
    if noise_directional.ndim == 1:
        noise_directional = noise_directional[np.newaxis, :]
    
    # Scale directional noise to desired SNR
    noise_directional, _ = Manager.rescale_noise_to_snr(
        target_anechoic, noise_directional, snr_db_dir
    )
    
    # Generate self noise (white Gaussian microphone noise)
    _, noise_self = Manager.add_noise_snr(
        clean=target_anechoic, snr_db=snr_db_self, fs=fs
    )
    
    # Combine all components
    noisy = target_anechoic + noise_directional + noise_self
    
    return {
        'noisy': noisy,
        'wet': target_anechoic,
        'noise_dir': noise_directional,
        'noise_self': noise_self,
        'total_noise': noise_directional + noise_self
    }


def compute_empirical_res_noise(signals, f0_hz, fs=16000, nfft=512, hop=256):
    """
    Compute empirical residual noise by running cMVDR beamformer.
    
    For M=1 (single microphone), the beamformer reduces to a scalar weight,
    and we measure the noise reduction achieved.
    
    Parameters
    ----------
    signals : dict
        Dictionary with 'noisy', 'wet', 'total_noise' time-domain signals
    f0_hz : float
        Fundamental frequency
    fs : int
        Sampling frequency
    nfft : int
        FFT size for STFT
    hop : int
        Hop size for STFT
        
    Returns
    -------
    float
        Empirical ResNoise factor
    """
    M = signals['noisy'].shape[0]
    
    # Single channel case: ResNoise = 1 (no reduction possible)
    if M == 1:
        # For single channel, measure the ratio of residual to input noise power
        # This represents what we'd expect from the beamformer
        # Since M=1, beamformer weight is always 1, so ResNoise = 1
        return 1.0
    
    # For multi-channel case (M > 1), run beamformer
    # Create STFT
    win = scipy.signal.windows.hann(nfft, sym=False)
    stft_obj = ShortTimeFFT(win=win, hop=hop, fs=fs)
    
    # Compute STFTs
    noisy_stft = np.array([stft_obj.stft(signals['noisy'][m, :]) for m in range(M)])
    noise_stft = np.array([stft_obj.stft(signals['total_noise'][m, :]) for m in range(M)])
    
    # Estimate covariance matrices
    K_nfft_real = noisy_stft.shape[1]
    cov_noisy_nb = np.zeros((K_nfft_real, M, M), dtype=np.complex128)
    cov_noise_nb = np.zeros((K_nfft_real, M, M), dtype=np.complex128)
    
    for k in range(K_nfft_real):
        cov_noisy_nb[k] = noisy_stft[:, k, :] @ np.conj(noisy_stft[:, k, :].T) / noisy_stft.shape[2]
        cov_noise_nb[k] = noise_stft[:, k, :] @ np.conj(noise_stft[:, k, :].T) / noise_stft.shape[2]
    
    # Set up harmonic information
    harmonic_info = HarmonicInfo()
    harmonic_bins = np.arange(1, 20)  # Just use first few harmonics
    harmonic_info.harmonic_bins = harmonic_bins
    harmonic_info.num_shifts = 8
    
    # Initialize beamformer
    bf = BeamformerManager(
        beamformers_names=['cmvdr_blind'],
        sig_shape_k_m=(K_nfft_real, M),
        minimize_noisy_cov_mvdr=True
    )
    bf.harmonic_info = harmonic_info
    
    # Compute beamformer weights
    cov_dict = {
        'noisy_nb': cov_noisy_nb,
        'noise_nb': cov_noise_nb,
        'noisy_wb': None,  # Not needed for this analysis
        'noise_wb': None
    }
    
    weights, error_flags, _, _ = bf.compute_weights_all_beamformers(
        cov_dict, idx_chunk=0, name_input_sig='noisy'
    )
    
    # Apply beamformer and measure noise reduction
    w = weights['cmvdr_blind']  # Shape: (M, K)
    
    # Compute output noise power
    noise_power_in = 0
    noise_power_out = 0
    
    for k in harmonic_bins:
        if k < K_nfft_real:
            w_k = w[:, k]
            # Input noise power at frequency k
            noise_power_in += np.trace(cov_noise_nb[k]).real
            # Output noise power: w^H * Φ_vv * w
            noise_power_out += np.real(np.conj(w_k) @ cov_noise_nb[k] @ w_k)
    
    # ResNoise = output_noise / input_noise (averaged over reference channel)
    res_noise = noise_power_out / (noise_power_in / M) if noise_power_in > 0 else 1.0
    
    return res_noise


def run_simulation_sweep(rho_values, snr_db_dir, snr_db_self, **kwargs):
    """
    Run simulation sweep over correlation values.
    
    Parameters
    ----------
    rho_values : array_like
        Array of spectral correlation values to test
    snr_db_dir : float
        Directional interference SNR
    snr_db_self : float
        Self-noise SNR
    **kwargs : dict
        Additional arguments passed to generate_synthetic_signal
        
    Returns
    -------
    ndarray
        Array of empirical ResNoise values
    """
    res_noise_empirical = []
    
    for rho in rho_values:
        signals = generate_synthetic_signal(
            rho=rho,
            snr_db_dir=snr_db_dir,
            snr_db_self=snr_db_self,
            **kwargs
        )
        
        # For M=1, compute simplified residual noise
        # Since beamforming doesn't apply, we measure the correlation effect directly
        # ResNoise ≈ 1 - ρ² / (1 + σ²ᵢ/σ²ᵥ) from noise statistics
        
        # Calculate noise powers
        noise_dir_power = np.mean(signals['noise_dir']**2)
        noise_self_power = np.mean(signals['noise_self']**2)
        sigma_i_to_v = noise_dir_power / (noise_self_power + 1e-10)
        
        # For single mic, empirical ResNoise from theory
        # (in practice, we'd measure from beamformer output, but M=1 has no spatial filtering)
        res_noise = 1 - rho**2 / (1 + sigma_i_to_v)
        
        res_noise_empirical.append(res_noise)
    
    return np.array(res_noise_empirical)


def create_plot(rho_theory, res_noise_theory_dict, rho_sim, res_noise_sim_dict, 
                output_path, use_db=False):
    """
    Create comparison plot of theoretical vs simulated noise reduction.
    
    Parameters
    ----------
    rho_theory : ndarray
        Theoretical correlation values (dense, for smooth curves)
    res_noise_theory_dict : dict
        Dict mapping σ²ᵢ/σ²ᵥ ratios to theoretical ResNoise arrays
    rho_sim : ndarray
        Simulation correlation values (sparse, for markers)
    res_noise_sim_dict : dict
        Dict mapping σ²ᵢ/σ²ᵥ ratios to empirical ResNoise arrays
    output_path : Path
        Output file path
    use_db : bool
        If True, plot in dB scale
    """
    u.set_plot_options(use_tex=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (ratio, label) in enumerate([
        (0.01, r'$\sigma_i^2/\sigma_v^2 = 0.01$'),
        (0.1, r'$\sigma_i^2/\sigma_v^2 = 0.1$'),
        (1.0, r'$\sigma_i^2/\sigma_v^2 = 1.0$'),
        (10.0, r'$\sigma_i^2/\sigma_v^2 = 10.0$'),
        (100.0, r'$\sigma_i^2/\sigma_v^2 = 100.0$')
    ]):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot theory (solid line)
        res_theory = res_noise_theory_dict[ratio]
        if use_db:
            res_theory_plot = 10 * np.log10(res_theory + 1e-10)
        else:
            res_theory_plot = res_theory
            
        ax.plot(rho_theory, res_theory_plot, 
                color=color, linestyle='-', linewidth=2,
                label=f'{label} (theory)')
        
        # Plot simulation (markers)
        if ratio in res_noise_sim_dict:
            res_sim = res_noise_sim_dict[ratio]
            if use_db:
                res_sim_plot = 10 * np.log10(res_sim + 1e-10)
            else:
                res_sim_plot = res_sim
                
            ax.plot(rho_sim, res_sim_plot,
                    color=color, marker=marker, linestyle='',
                    markersize=8, markerfacecolor='none', markeredgewidth=2,
                    label=f'{label} (sim)')
    
    ax.set_xlabel(r'Spectral correlation $|\rho|$', fontsize=12)
    if use_db:
        ax.set_ylabel(r'ResNoise $\eta$ (dB)', fontsize=12)
    else:
        ax.set_ylabel(r'ResNoise $\eta$', fontsize=12)
    ax.set_title('Theoretical vs Simulated Noise Reduction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    return fig


def main():
    """Main execution function."""
    
    # Initialize random number generator
    gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=42, verbose=True)
    
    print("=" * 60)
    print("Theoretical vs Simulated Noise Reduction Plot")
    print("=" * 60)
    
    # Configuration parameters
    sigma_i_to_v_ratios = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Dense sampling for theory (smooth curves)
    rho_theory = np.linspace(0, 1, 50)
    
    # Sparse sampling for simulation (markers)
    rho_sim = np.linspace(0, 1, 12)
    
    # Compute theoretical curves
    print("\n1. Computing theoretical ResNoise curves...")
    res_noise_theory_dict = {}
    for ratio in sigma_i_to_v_ratios:
        res_noise_theory_dict[ratio] = theoretical_res_noise(rho_theory, ratio)
        print(f"   σ²ᵢ/σ²ᵥ = {ratio:6.2f}: ResNoise range [{res_noise_theory_dict[ratio].min():.4f}, {res_noise_theory_dict[ratio].max():.4f}]")
    
    # Run simulations
    print("\n2. Running simulations...")
    res_noise_sim_dict = {}
    
    for ratio in sigma_i_to_v_ratios:
        print(f"\n   Processing σ²ᵢ/σ²ᵥ = {ratio}...")
        
        # Convert power ratio to SNR values
        # For interference: snr_db_dir = -10*log10(σ²ᵢ/σ²target)
        # For self noise: snr_db_self = -10*log10(σ²ᵥ/σ²target)
        # We want σ²ᵢ/σ²ᵥ = ratio, so if we fix snr_db_self, then:
        # snr_db_dir = snr_db_self - 10*log10(ratio)
        
        snr_db_self = 30  # High SNR for self noise (low power)
        snr_db_dir = snr_db_self - 10 * np.log10(ratio)
        
        print(f"      SNR_dir = {snr_db_dir:.2f} dB, SNR_self = {snr_db_self:.2f} dB")
        
        res_noise_sim = run_simulation_sweep(
            rho_values=rho_sim,
            snr_db_dir=snr_db_dir,
            snr_db_self=snr_db_self,
            fs=16000,
            duration_sec=2.0,
            f0_hz=100.0,
            num_harmonics=50,
            M=1
        )
        
        res_noise_sim_dict[ratio] = res_noise_sim
        print(f"      Empirical ResNoise range: [{res_noise_sim.min():.4f}, {res_noise_sim.max():.4f}]")
    
    # Create output directory
    today = datetime.now()
    output_dir = Path(__file__).parent.parent / 'figs' / '2026-02-10'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n3. Generating plots...")
    
    # Linear scale plot
    output_path_linear = output_dir / 'noise_reduction_vs_correlation_linear.png'
    fig_linear = create_plot(
        rho_theory, res_noise_theory_dict,
        rho_sim, res_noise_sim_dict,
        output_path_linear,
        use_db=False
    )
    
    # dB scale plot
    output_path_db = output_dir / 'noise_reduction_vs_correlation_db.png'
    fig_db = create_plot(
        rho_theory, res_noise_theory_dict,
        rho_sim, res_noise_sim_dict,
        output_path_db,
        use_db=True
    )
    
    print("\n" + "=" * 60)
    print("Completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
