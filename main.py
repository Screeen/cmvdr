import time
import subprocess
import argparse
from cmvdr.util import config, globs as gs
from cmvdr.experiment_manager import ExperimentManager

# Main entry point for the cyclic beamforming experiment
# This script sets up the experiment, runs it, and manages the output
# Read from command line arguments to configure the experiment
# The script also handles the configuration loading and random number generation setup
if __name__ == '__main__':

    argparse.ArgumentParser(
        description="Run cyclic beamforming experiment with specified configurations."
    )
    parser = argparse.ArgumentParser(description="Cyclic Beamforming Experiment Runner")

    # Specify the path or name of the configuration file
    parser.add_argument('-c', '--config', type=str, default='synthetic_demo.yaml',
                        help='Path to the configuration file (YAML format). Default is synthetic_demo.yaml',
                        metavar='CONFIG_FILE')
    args = parser.parse_args()

    cfg_original = config.load_configuration_outer(args.config)
    gs.rng, cfg_original['seed_extracted'] = gs.compute_rng(cfg_original['seed_is_random'],
                                                            cfg_original['seed_if_not_random'])

    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    res = ExperimentManager.run_experiment(cfg_original)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s")

    # Open the folder with the figures if the elapsed time is more than 60 seconds or if there are many simulations
    if elapsed_time > 60 or res['cfg_original']['num_montecarlo_simulations'] > 10:
        # Use 'open' command for macOS or 'xdg-open' for Linux
        # To detach, we use stdout and stderr redirection to DEVNULL
        subprocess.Popen(["open", res['target_path_figs']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
