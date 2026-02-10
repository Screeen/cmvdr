"""
Plot theoretical vs simulated noise reduction factor (η).

This script validates the theoretical noise reduction performance from equation 
eq:res_noise_cmvdr:factor against simulation results using the cMVDR/cMPDR beamformer.

Theoretical formula: η = 1 - ρ² / (1 + σ²ᵢ/σ²ᵥ)
where:
- ρ is the spectral correlation coefficient between harmonic components
- σ²ᵢ is the power of one harmonic component
- σ²ᵥ is the power of another harmonic component (σ²ᵥ > 0 always)

The ratio σ²ᵢ/σ²ᵥ represents the relative power between different harmonic components.

This script uses synthetic covariance matrices with equicorrelated structure
to directly test the beamformer performance without complex signal generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cmvdr.beamforming.cyclic_mvdr import CyclicMVDR
from cmvdr.util.harmonic_info import HarmonicInfo
from cmvdr.util import utils as u
from cmvdr.util import plotter
from cmvdr.util import globs as gs


def generate_equicorrelated_covariance(rho, P, noise_power=1.0):
    """
    Generate equicorrelated covariance matrix for cyclic processing.
    
    This creates a covariance matrix where all off-diagonal elements have
    correlation ρ, simulating the structure of harmonically-related noise.
    
    Parameters
    ----------
    rho : float
        Correlation coefficient between frequency-shifted components (0 to 1)
    P : int
        Number of cyclic shifts (virtual channels)
    noise_power : float
        Total noise power
        
    Returns
    -------
    ndarray
        (P, P) equicorrelated covariance matrix
    """
    # Create equicorrelated structure: diagonal = 1, off-diagonal = rho
    cov = np.ones((P, P)) * rho
    np.fill_diagonal(cov, 1.0)
    
    # Scale by noise power
    cov = cov * noise_power
    
    return cov


def compute_theoretical_eta(rho, sigma_i_to_v_ratio=1.0):
    """
    Calculate theoretical residual noise factor η.
    
    Formula: η = 1 - ρ² / (1 + σ²ᵢ/σ²ᵥ)
    
    Parameters
    ----------
    rho : float or ndarray
        Spectral correlation coefficient |ρ| (0 to 1)
    sigma_i_to_v_ratio : float
        Power ratio σ²ᵢ/σ²ᵥ between harmonic components (default: 1.0, must be > 0)
        
    Returns
    -------
    float or ndarray
        Residual noise factor η
    """
    return 1 - rho**2 / (1 + sigma_i_to_v_ratio)


def compute_empirical_eta_single_bin(rho, P, noise_power=1.0):
    """
    Compute empirical η using cMVDR beamformer on a single frequency bin.
    
    This function:
    1. Creates synthetic covariance matrices with equicorrelated structure
    2. Runs cMVDR beamformer to get optimal weights
    3. Computes output noise power: η = w^H * Φ_n * w / noise_power
    
    Parameters
    ----------
    rho : float
        Target spectral correlation (0 to 1)
    P : int
        Number of cyclic shifts (virtual channels)
    noise_power : float
        Input noise power per component
        
    Returns
    -------
    float
        Empirical residual noise factor η
    """
    # For single microphone (M=1), single frequency bin (K=1)
    M = 1
    K = 1
    
    # Generate equicorrelated covariance matrices
    # Wideband (cyclic) covariance: (K, M*P, M*P)
    cov_noise_wb = np.zeros((K, M * P, M * P), dtype=np.complex128)
    cov_noise_wb[0] = generate_equicorrelated_covariance(rho, M * P, noise_power)
    
    # For the input signal covariance, assume it's white (uncorrelated) target
    # plus the equicorrelated noise
    target_power = 10.0  # Arbitrary target power (doesn't affect η)
    cov_noisy_wb = cov_noise_wb.copy()
    # Add white target signal contribution (only to diagonal)
    for i in range(M * P):
        cov_noisy_wb[0, i, i] += target_power
    
    # Narrowband covariances (fallback for non-cyclic bins)
    cov_noise_nb = np.zeros((K, M, M), dtype=np.complex128)
    cov_noise_nb[0] = np.eye(M) * noise_power
    
    cov_noisy_nb = np.zeros((K, M, M), dtype=np.complex128)
    cov_noisy_nb[0] = np.eye(M) * (noise_power + target_power)
    
    # Create covariance dictionary
    cov_dict = {
        'noisy_wb': cov_noisy_wb,
        'noise_wb': cov_noise_wb,
        'noisy_nb': cov_noisy_nb,
        'noise_nb': cov_noise_nb
    }
    
    # Initialize beamformer
    loadings_cfg = (0, 0, 1000)  # (min, max, max_condition_number)
    cm = CyclicMVDR(loadings_cfg, (K, M), minimize_noisy_cov_mvdr=True)
    
    # Set up harmonic info (all bins are cyclic with P shifts)
    cm.harmonic_info = SimpleHarmonicInfo(K, P_all=np.array([P]))
    
    # Compute beamformer weights
    cyclic_bins = np.array([0])  # Only one bin, and it's cyclic
    P_all = np.array([P])
    
    try:
        weights, err_flags, cond_num_cov, sv = cm.compute_cyclic_mvdr_beamformers(
            cov_dict, 'blind', cyclic_bins, P_all=P_all
        )
        
        # Extract weights for the single frequency bin
        w_c = weights[:, 0]  # Shape: (M*P,)
        
        # Compute output noise power: w^H * Φ_n * w
        output_noise_power = np.real(np.conj(w_c) @ cov_noise_wb[0] @ w_c)
        
        # Input noise power (average over M*P channels)
        input_noise_power = noise_power
        
        # Residual noise factor: η = output_noise / input_noise
        eta = output_noise_power / input_noise_power
        
        return eta
        
    except Exception as e:
        print(f"Warning: Beamformer computation failed for rho={rho}: {e}")
        return 1.0  # Return no noise reduction if failed


class SimpleHarmonicInfo:
    """Minimal harmonic_info for testing."""
    def __init__(self, K, P_all=None):
        self.K = K
        if P_all is None:
            self._P_all = np.ones(K, dtype=int)
        else:
            self._P_all = np.array(P_all, dtype=int)
    
    def get_num_shifts_all_frequencies(self):
        return self._P_all


def run_simulation_sweep(rho_values, P=8, noise_power=1.0):
    """
    Run simulation sweep over correlation values.
    
    Parameters
    ----------
    rho_values : array_like
        Array of spectral correlation values to test
    P : int
        Number of cyclic shifts (virtual channels)
    noise_power : float
        Input noise power
        
    Returns
    -------
    ndarray
        Array of empirical η values
    """
    eta_empirical = []
    
    for rho in rho_values:
        eta = compute_empirical_eta_single_bin(rho, P, noise_power)
        eta_empirical.append(eta)
    
    return np.array(eta_empirical)


def create_plot_multi(rho_theory, eta_theory_dict, sigma_ratios, rho_sim, eta_sim, output_path, use_db=False):
    """
    Create comparison plot with multiple theoretical curves for different σ²ᵢ/σ²ᵥ ratios.
    
    Parameters
    ----------
    rho_theory : ndarray
        Theoretical correlation values (dense, for smooth curves)
    eta_theory_dict : dict
        Dict mapping σ²ᵢ/σ²ᵥ ratios to theoretical η arrays
    sigma_ratios : list
        List of σ²ᵢ/σ²ᵥ ratios to plot
    rho_sim : ndarray
        Simulation correlation values (sparse, for markers)
    eta_sim : ndarray
        Empirical η values from simulation (for σ²ᵢ/σ²ᵥ = 1.0)
    output_path : Path
        Output file path
    use_db : bool
        If True, plot in dB scale
    """
    # Check if LaTeX is available
    use_latex = plotter.is_tex_plotting_available()
    if not use_latex:
        print("   LaTeX not available, using standard matplotlib rendering")
    
    # Set plot options based on LaTeX availability
    u.set_plot_options(use_tex=use_latex)
    
    # Use proper figure size for double-column LaTeX documents
    width = u.get_plot_width_double_column_latex()
    height = width * 0.75  # Aspect ratio
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Color scheme for different ratios
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    # Plot theory curves for different σ²ᵢ/σ²ᵥ ratios
    for idx, ratio in enumerate(sigma_ratios):
        eta_theory = eta_theory_dict[ratio]
        
        if use_db:
            eta_theory_plot = 10 * np.log10(eta_theory + 1e-10)
        else:
            eta_theory_plot = eta_theory
        
        color = colors[idx % len(colors)]
        
        if use_latex:
            label = r'$\sigma_i^2/\sigma_v^2 = ' + f'{ratio:.1f}' + r'$'
        else:
            label = f'σ²ᵢ/σ²ᵥ = {ratio:.1f}'
        
        ax.plot(rho_theory, eta_theory_plot, 
                color=color, linestyle='-', linewidth=1.5,
                label=label)
    
    # Plot simulation (markers) for σ²ᵢ/σ²ᵥ = 1.0
    if use_db:
        eta_sim_plot = 10 * np.log10(eta_sim + 1e-10)
        ylabel = r'$\eta$ (dB)' if use_latex else 'η (dB)'
    else:
        eta_sim_plot = eta_sim
        ylabel = r'$\eta$' if use_latex else 'η'
    
    ax.plot(rho_sim, eta_sim_plot,
            color='black', marker='o', linestyle='',
            markersize=4, markerfacecolor='none', markeredgewidth=1.0,
            label='Simulation (cMVDR)')
    
    ax.set_xlabel(r'Spectral correlation $|\rho|$' if use_latex else 'Spectral correlation |ρ|')
    ax.set_ylabel(ylabel)
    
    if use_latex:
        ax.set_title(r'Noise Reduction Factor $\eta$ vs.\ Correlation')
    else:
        ax.set_title('Noise Reduction Factor η vs. Correlation')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Figure saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"   PDF saved to: {pdf_path}")
    
    plt.close(fig)
    
    return fig


def create_plot(rho_theory, eta_theory, rho_sim, eta_sim, output_path, use_db=False):
    """
    Create comparison plot of theoretical vs simulated noise reduction.
    
    Parameters
    ----------
    rho_theory : ndarray
        Theoretical correlation values (dense, for smooth curves)
    eta_theory : ndarray
        Theoretical η values
    rho_sim : ndarray
        Simulation correlation values (sparse, for markers)
    eta_sim : ndarray
        Empirical η values from simulation
    output_path : Path
        Output file path
    use_db : bool
        If True, plot in dB scale
    """
    # Check if LaTeX is available
    use_latex = plotter.is_tex_plotting_available()
    if not use_latex:
        print("   LaTeX not available, using standard matplotlib rendering")
    
    # Set plot options based on LaTeX availability
    u.set_plot_options(use_tex=use_latex)
    
    # Use proper figure size for double-column LaTeX documents
    width = u.get_plot_width_double_column_latex()
    height = width * 0.75  # Aspect ratio
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot theory (solid line)
    if use_db:
        eta_theory_plot = 10 * np.log10(eta_theory + 1e-10)
        eta_sim_plot = 10 * np.log10(eta_sim + 1e-10)
        ylabel = r'$\eta$ (dB)' if use_latex else 'η (dB)'
    else:
        eta_theory_plot = eta_theory
        eta_sim_plot = eta_sim
        ylabel = r'$\eta$' if use_latex else 'η'
    
    ax.plot(rho_theory, eta_theory_plot, 
            color='tab:blue', linestyle='-', linewidth=1.5,
            label='Theory')
    
    # Plot simulation (markers)
    ax.plot(rho_sim, eta_sim_plot,
            color='tab:orange', marker='o', linestyle='',
            markersize=6, markerfacecolor='none', markeredgewidth=1.5,
            label='Simulation (cMVDR)')
    
    ax.set_xlabel(r'Spectral correlation $|\rho|$' if use_latex else 'Spectral correlation |ρ|')
    ax.set_ylabel(ylabel)
    
    if use_latex:
        ax.set_title(r'Noise Reduction Factor $\eta$ vs.\ Correlation')
    else:
        ax.set_title('Noise Reduction Factor η vs. Correlation')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Figure saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"   PDF saved to: {pdf_path}")
    
    plt.close(fig)
    
    return fig


def main():
    """Main execution function."""
    
    # Initialize random number generator
    gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=42, verbose=True)
    
    print("=" * 60)
    print("Noise Reduction Factor η vs Correlation")
    print("=" * 60)
    
    # Configuration parameters
    P = 8  # Number of cyclic shifts (virtual channels)
    noise_power = 1.0
    
    # Test multiple σ²ᵢ/σ²ᵥ ratios (power ratios between harmonic components)
    sigma_ratios = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Dense sampling for theory (smooth curves)
    rho_theory = np.linspace(0, 1, 100)
    
    # Sparse sampling for simulation (markers)
    rho_sim = np.linspace(0, 1, 15)
    
    # Compute theoretical curves for different ratios
    print("\n1. Computing theoretical η curves for different σ²ᵢ/σ²ᵥ ratios...")
    eta_theory_dict = {}
    for ratio in sigma_ratios:
        eta_theory_dict[ratio] = compute_theoretical_eta(rho_theory, sigma_i_to_v_ratio=ratio)
        print(f"   σ²ᵢ/σ²ᵥ = {ratio:4.1f}: η range [{eta_theory_dict[ratio].min():.4f}, {eta_theory_dict[ratio].max():.4f}]")
    
    # Run simulations for one representative case (σ²ᵢ/σ²ᵥ = 1.0)
    print("\n2. Running cMVDR simulations for σ²ᵢ/σ²ᵥ = 1.0...")
    print(f"   Using P={P} cyclic shifts")
    
    eta_sim = run_simulation_sweep(rho_sim, P=P, noise_power=noise_power)
    print(f"   Simulation: η range [{eta_sim.min():.4f}, {eta_sim.max():.4f}]")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figs' / '2026-02-10'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n3. Generating plots...")
    
    # Linear scale plot with multiple theory curves
    output_path_linear = output_dir / 'eta_vs_correlation_multi_linear.png'
    fig_linear = create_plot_multi(
        rho_theory, eta_theory_dict, sigma_ratios,
        rho_sim, eta_sim,
        output_path_linear,
        use_db=False
    )
    
    # dB scale plot with multiple theory curves
    output_path_db = output_dir / 'eta_vs_correlation_multi_db.png'
    fig_db = create_plot_multi(
        rho_theory, eta_theory_dict, sigma_ratios,
        rho_sim, eta_sim,
        output_path_db,
        use_db=True
    )
    
    print("\n" + "=" * 60)
    print("Completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
