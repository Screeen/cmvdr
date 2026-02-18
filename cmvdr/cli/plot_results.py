"""
CLI tool for regenerating plots from saved experiment results.

Usage:
    cmvdr-plot -p exp_results/2026-02-13/11h28
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime

from cmvdr.util import config, plotter as pl, utils as u


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from saved experiment results.",
        epilog="Example: cmvdr-plot -p exp_results/2026-02-13/11h28"
    )
    parser.add_argument(
        "-p", "--path",
        required=True,
        type=str,
        help="Path to the experiment results directory (e.g., exp_results/2026-02-13/11h28)"
    )

    args = parser.parse_args()

    # Parse and validate paths
    exp_path = Path(args.path)
    if not exp_path.exists():
        print(f"Error: Path does not exist: {exp_path}")
        return

    data_path = exp_path / 'data'
    results_pkl_path = data_path / 'results_beamforming.pkl'
    config_yaml_path = exp_path / 'config.yaml'

    if not results_pkl_path.exists():
        print(f"Error: Results file not found: {results_pkl_path}")
        return

    if not config_yaml_path.exists():
        print(f"Error: Config file not found: {config_yaml_path}")
        return

    # Load results and config
    print(f"Loading results from {results_pkl_path}...")
    with open(results_pkl_path, 'rb') as f:
        results_data_type_plots = pickle.load(f)

    print(f"Loading configuration from {config_yaml_path}...")
    cfg = config.load_yaml_from_path(config_yaml_path)

    # Reconstruct plot settings
    plot_sett = config.ConfigManager.get_plot_settings(cfg['plot'])

    # Create timestamped output folder for new figures
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
    target_path_figs = exp_path / f'figs-{timestamp}'
    target_path_figs.mkdir(parents=True, exist_ok=True)

    print(f"Regenerating plots in {target_path_figs}...")

    # Set plot options
    tex_available = pl.is_tex_plotting_available(plot_sett['force_no_tex'])
    u.set_plot_options(use_tex=plot_sett['use_tex'] and tex_available)

    # Regenerate plots
    try:
        pl.visualize_all_results(
            results_data_type_plots,
            plot_sett,
            cfg,
            plot_db=False,
            print_summary=False,
            target_path_figs_=target_path_figs,
            print_full_table=True
        )

        # Move *.pkl files to figs_pkl subdirectory
        if target_path_figs.exists() and any(target_path_figs.iterdir()):
            target_path_figs_pkl = target_path_figs / 'figs_pkl'
            target_path_figs_pkl.mkdir(parents=True, exist_ok=True)
            for pkl_file in target_path_figs.glob('*.pkl'):
                pkl_file.rename(target_path_figs_pkl / pkl_file.name)

        print(f"Plots successfully regenerated in {target_path_figs}")

    except Exception as exc:
        print(f"Error during plotting: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

