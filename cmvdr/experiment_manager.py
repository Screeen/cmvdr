import numpy as np
from scipy.signal import ShortTimeFFT
import time
from copy import deepcopy as dcopy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from cmvdr import config
from cmvdr.data_generator import DataGenerator
from cmvdr import (covariance_estimator,
                                   manager)
from cmvdr.f0_manager import F0ChangeAmount
from cmvdr.harmonic_info import HarmonicInfo
from cmvdr.modulator import Modulator
from cmvdr.coherence_manager import CoherenceManager
from cmvdr.beamformer_manager import BeamformerManager
from cmvdr import (utils as u, data_generator, evaluator, plotter as pl, f0_manager)
from cmvdr.player import Player  # do not remove, useful for quick evaluation of signals from cmd line
from cmvdr import audio_disk_loader as audio_loader

u.set_printoptions_numpy()
threshold_hz_f0_std = np.inf


class ExperimentManager:
    def __init__(self):
        pass

    @staticmethod
    def run_cov_estimation_beamforming(signals, f0man, f0_over_time, harmonic_freqs_est, cfg, dft_props, do_plots=False,
                                       SFT: ShortTimeFFT = None, name_input_sig='noisy'):

        def print_log_chunks():
            if do_plots:
                slice_cov_est = slice_cov_est_list[idx_chunk]
                if num_chunks > 1:
                    print(f"Process chunk by chunk. Chunk {idx_chunk + 1}/{num_chunks}, "
                          f"frames {slice_cov_est.start}-{slice_cov_est.stop}/{L1_frames_all_chunks}.")
                else:
                    print(f"Process all chunks at once. {num_chunks = }, {L1_frames_all_chunks = }, signal duration = "
                          f"{L1_frames_all_chunks * SFT.delta_t:.2f}s.")

        def print_log_noise():
            if do_plots:
                print(f"Noise signals generated. "
                      f"Total duration is {int(cfg['cov_estimation']['noise_cov_est_len_seconds'] * SFT.fs) / SFT.fs:.2f}s.")

        # cov_dict_prev = covariance_estimator.CovarianceHolder()  # empty covariance holder
        cov_dict_prev = {}
        coh_man = CoherenceManager()
        m = manager.Manager()
        f0_tracker = f0_manager.F0Tracker()

        K_nfft_real = dft_props['nfft_real']
        is_cmwf_bf = cfg['cyclostationary_target']
        use_pseudo_cov = any(['wl' in x for x in cfg['beamforming']['methods']])
        ce = covariance_estimator.CovarianceEstimator(cfg['cov_estimation'], cfg['cyclostationary_target'],
                                                      subtract_mean=False, use_pseudo_cov=use_pseudo_cov)
        M = signals[name_input_sig]['stft'].shape[0]
        # use_masked_stft_for_evaluation = cfg['use_masked_stft_for_evaluation']

        target_rtf = np.array([])
        if 'wet' in signals:
            target_rtf = DataGenerator.calculate_ground_truth_rtf(signals['wet'])

        print_log_noise()

        # if np.isfinite(cfg['est_error_snr']):
        #     signals_unproc_with_est_noise = m.add_noise_to_signals(dcopy(signals_unproc),
        #                                                            cfg['est_error_snr'],
        #                                                            which_signals=['wet_rank1'])

        L1_frames_all_chunks = signals[name_input_sig]['stft'].shape[-1]
        bfd_all_chunks_stft, bfd_all_chunks_stft_masked = m.allocate_beamformed_signals(
            L1_frames_all_chunks, K_nfft_real, cfg['beamforming']['methods'])

        slice_bf_list, slice_cov_est_list, num_chunks = \
            (m.get_chunks_slices(L1_frames_all_chunks, dft_props=dft_props, time_props=cfg['time'],
                                 recursive_average=cfg['cov_estimation']['recursive_average']))

        bf = BeamformerManager(beamformers_names=cfg['beamforming']['methods'],
                               sig_shape_k_m=(K_nfft_real, M),
                               minimize_noisy_cov_mvdr=cfg['beamforming']['minimize_noisy_cov_mvdr'],
                               loadings=cfg['beamforming']['loadings'],
                               noise_var_rtf=cfg['noise']['noise_var_rtf'])
        mod_amount_numeric_list = []
        harm_info = HarmonicInfo()
        num_voiced_chunks = 0
        cyclic_bins_ratio_list = []

        # Loop over the chunks of the signal to estimate the covariance matrices and beamform the signals
        for idx_chunk in range(num_chunks):
            if not bf.beamformers_names:
                continue

            is_first_chunk = idx_chunk == 0
            slice_bf = slice_bf_list[idx_chunk]
            print_log_chunks()

            # Execute if we rely on f0 estimate instead of estimate of all harmonics (f0 only gives worse results)
            if 'f0' in cfg['harmonics_est']['algo']:
                harmonic_freqs_est = f0man.get_harmonics_as_multiples_of_f0(f0_over_time[slice_bf],
                                                                            cfg['cyclic']['freq_range_cyclic'])

            mod_amount = F0ChangeAmount.no_change
            if is_cmwf_bf or is_first_chunk:
                if cfg['cyclic']['use_global_coherence']:
                    harm_info, mod_amount = f0man.compute_harmonic_and_modulation_sets_global_coherence(
                        signals[cfg['cyclic']['coherence_source_signal_name']],
                        harmonic_freqs_est, SFT, cfg['cyclic'])
                else:
                    harm_info, mod_amount = f0man.compute_harmonic_and_modulation_sets_distance_based(
                        harmonic_freqs_est, cfg['harmonics_est'], cfg['cyclic'], dft_props, idx_chunk,
                        f0_over_time[slice_bf], f0_tracker)

                if is_first_chunk:
                    print(f"Num shifts per set: {harm_info.num_shifts_per_set}")

                # Collect statistics about the cyclic bins ratio
                if harm_info.alpha_mods_sets[0].size > 1:
                    num_voiced_chunks = num_voiced_chunks + 1
                    cyclic_bins_ratio = (np.sum(harm_info.mask_harmonic_bins) /
                                         max(1, harm_info.mask_harmonic_bins.size))
                    cyclic_bins_ratio_list.append(cyclic_bins_ratio)

                ce.harmonic_info = harm_info
                bf.harmonic_info = harm_info

            signals_to_modulate = config.ConfigManager.choose_signals_to_modulate(
                is_cmwf_bf, cfg['beamforming']['minimize_noisy_cov_mvdr'],
                is_first_chunk, cfg['cov_estimation']['recursive_average'], name_input_sig,
                skip_noise_cov_est=cfg['data_type'] == 'inference')

            # Re-modulate the signals with new harmonic info
            if mod_amount == F0ChangeAmount.small:
                signals = Modulator.modulate_signals(signals, signals_to_modulate, SFT,
                                                     harm_info.alpha_mods_sets, cfg['cyclic']['P_max'],
                                                     name_input_sig)

            ce.set_dimensions((K_nfft_real, M, cfg['cyclic']['P_max']))
            if is_first_chunk:
                print(f"CovarianceEstimator: set (K_nfft, M, P) to {ce.sig_shape_k_m_p}")

            cov_dict = ce.estimate_covariances(slice_cov_est_list[idx_chunk], signals, cov_dict_prev,
                                               num_mics_changed=ce.sig_shape_k_m_p[1] != M,
                                               modulation_amount=mod_amount,
                                               name_input_sig=name_input_sig)
            cov_dict_prev = dcopy(cov_dict)

            # Rough SNR estimate
            # noisy_noise = np.mean(np.trace(cov_dict['noisy_nb'], axis1=1, axis2=2).real)/np.mean(np.trace(cov_dict['noise_nb'], axis1=1, axis2=2).real)
            # print(f"{noisy_noise = }")

            harm_thr = coh_man.get_adaptive_harmonic_threshold(cfg['cyclic']['harmonic_threshold'], idx_chunk)
            local_coherence_selection = (harm_thr > 0 and idx_chunk % 20 == 0
                                         and not cfg['cyclic']['use_global_coherence'] and not is_cmwf_bf)
            if local_coherence_selection:
                ch_noisy_wb = cov_dict[name_input_sig + '_wb']
                ch_noise_wb = cov_dict['noise_wb']
                ch_noisy_wb, signals, harm_info = coh_man.remove_uncorrelated_modulations_local_coherence(
                    cov_read=ch_noise_wb if 'noise' in cfg['cyclic']['coherence_source_signal_name'] else ch_noisy_wb,
                    cov_write_list=[ch_noisy_wb],
                    signals=signals, harm_info=harm_info, harmonic_threshold=harm_thr, group_by_set=False)

                ce.harmonic_info = harm_info
                bf.harmonic_info = harm_info
                print(f"{idx_chunk = } {harm_info.num_shifts_per_set}")

            # Compute weights for beamformers
            weights, error_flags = bf.compute_weights_all_beamformers(cov_dict=cov_dict, rtf_oracle=target_rtf,
                                                                      idx_chunk=idx_chunk,
                                                                      name_input_sig=name_input_sig)
            # weights = bf.use_old_weights_if_error(weights, weights_previous, error_flags)
            # weights = bf.make_weights_dict_symmetric_around_central_frequency(K_nfft, weights)
            # weights_previous = dcopy(weights)

            # Beamform the signals
            mod_amount_bf = mod_amount if idx_chunk != 0 else F0ChangeAmount.no_change
            bfd = bf.beamform_signals(signals[name_input_sig]['stft'],
                                      signals[name_input_sig]['mod_stft_3d'], slice_bf, weights,
                                      mod_amount=mod_amount_bf)
            mod_amount_numeric_list.append(mod_amount.to_number())

            # Store the beamformed signals for the current chunk in the 'all_chunks' dictionary
            for key in bfd.keys():
                if key != name_input_sig:
                    bfd_all_chunks_stft[key][:, slice_bf] = bfd[key]
                    # if use_masked_stft_for_evaluation:
                    #     bfd_all_chunks_stft_masked[key][harm_info.mask_harmonic_bins, slice_bf] = bfd[key][
                    #         harm_info.mask_harmonic_bins]

            # end of chunks loop

        # if num_voiced_chunks > 0:
        #     print(f"{num_voiced_chunks/num_chunks = }:.1f")
        #     print(f"{np.mean(np.asarray(cyclic_bins_ratio_list)):.2f}")

        bf.check_beamformed_signals_non_zero(bfd_all_chunks_stft, signals)

        # Plot waveforms and spectrograms
        if do_plots:
            bench = 'mwf_blind' if is_cmwf_bf else 'mvdr_blind'
            # debug_title = f"{parameter_to_vary} = {str(param_value)}",
            pl.plot_waveforms_and_spectrograms(dcopy(cfg['plot']), bfd_all_chunks_stft, dft_props,
                                               f0_tracker.alpha_smooth_list,
                                               num_chunks, slice_bf_list, dft_props['nfft'], SFT.delta_t,
                                               freq_max_cyclic=cfg['cyclic']['freq_range_cyclic'][1],
                                               benchmark_algo=bench)

        return bfd_all_chunks_stft

    @staticmethod
    def convert_signals_time_domain(bfd_all_chunks_stft, SFT_real):
        # Compute the time-domain signals from the beamformed STFTs and append to dict
        signals_bfd_dict = {}
        for key in bfd_all_chunks_stft.keys():
            signals_bfd_dict[key] = {'time': SFT_real.istft(bfd_all_chunks_stft[key]).real,
                                     'stft': bfd_all_chunks_stft[key],
                                     # 'stft_masked': bfd_all_chunks_stft_masked[key],
                                     'display_name': pl.get_display_name(key)}

        return signals_bfd_dict

    @classmethod
    def apply_post_filtering(cls, signals, signals_bfd_dict, dft_props, f0man, f0_over_time, harmonic_freqs_est,
                             do_plots, SFT, SFT_real, cfg_original):
        """ Apply post-filtering to the beamformed signals. """
        cfg_copy = dcopy(cfg_original)

        og_fields = ['noise_cov_est', 'wet_rank1']
        for f in og_fields:  # these signals are not beamformed, so we need to add them to the dict
            signals_bfd_dict[f] = signals[f]

        # Broadcast the signals
        for key in signals_bfd_dict.keys():
            signals_bfd_dict[key]['time'] = np.atleast_2d(signals_bfd_dict[key]['time'])
            if signals_bfd_dict[key]['stft'].ndim == 2:
                signals_bfd_dict[key]['stft'] = signals_bfd_dict[key]['stft'][np.newaxis]
                if 'stft_conj' in signals_bfd_dict[key].keys():
                    signals_bfd_dict[key]['stft_conj'] = signals_bfd_dict[key]['stft_conj'][np.newaxis]

        # Make signals mono
        for f in og_fields:
            for key in signals_bfd_dict[f].keys():
                signals_bfd_dict[f][key] = signals_bfd_dict[f][key][0:1]
        signals_bfd_dict['mvdr_blind']['stft_conj'] = np.conj(signals_bfd_dict['mvdr_blind']['stft'])

        cfg_copy['M'] = 1
        cfg_copy['beamforming']['methods'] = ['cmvdr_blind']
        bfd_all_chunks_stft_2 = ExperimentManager.run_cov_estimation_beamforming(
            signals_bfd_dict, f0man, f0_over_time, harmonic_freqs_est, cfg_copy, dft_props, do_plots, SFT,
            name_input_sig='mvdr_blind')

        bfd_all_chunks_stft_2_ = {'pf' + key: value for key, value in bfd_all_chunks_stft_2.items()}
        signals_bfd_dict = ExperimentManager.convert_signals_time_domain(bfd_all_chunks_stft_2_, SFT_real)

        return signals_bfd_dict

    @staticmethod
    def run_experiment(cfg_original, args):
        """ Run the cyclic beamforming experiment."""

        # Load the configuration and merge with secondary config based on data type
        if args.data_type != 'config':
            cfg_original.update({'data_type': args.data_type})

        cfg_original = config.load_and_merge_secondary_config(cfg_original, cfg_original['data_type'])
        results_data_type_plots = {}
        results_data_type_freq_est_plots = {}
        signals_dict_all_variations_time = {}

        varying_parameters_names = config.ConfigManager.get_varying_parameters_names(cfg_original)
        plot_sett = config.ConfigManager.get_plot_settings(cfg_original['plot'])
        u.set_plot_options(use_tex=plot_sett['use_tex'])

        # quick check to make sure all parameters are valid to avoid errors later
        [config.get_varying_param_values(dcopy(cfg_original), x) for x in varying_parameters_names]

        for parameter_to_vary in varying_parameters_names:
            # parameter_to_vary could be 'P_max', 'f0_err_percent', SNR, etc.
            # alpha_hz_printed_already = False
            # weights_previous = {}

            cfg_default = dcopy(cfg_original)
            varying_param_values = config.get_varying_param_values(cfg_default, parameter_to_vary)
            fs = cfg_default['fs']
            cfg_default['target'] = config.update_target_settings(cfg_default['target'])
            config.check_cyclic_target_or_not(cfg_default)
            cfg_default['beamforming']['methods'] = BeamformerManager.infer_beamforming_methods(
                cfg_default['beamforming'])

            results_dict = {str(varying_param_value): [] for varying_param_value in varying_param_values}
            results_freq_est = dcopy(results_dict)
            signals_dict_all_variations_time = {str(param_val): {} for param_val in varying_param_values}

            signals_bfd_dict_backup = {}
            post_filtering = False

            # Each iteration is a different value of the parameter to vary. E.g., SNR = 0dB, then 10dB, etc.
            for idx_var, param_value in enumerate(varying_param_values):
                print(f"Varying parameter {parameter_to_vary}. Current value: {parameter_to_vary} = {str(param_value)}")
                cfg = config.get_config_single_variation(cfg_default, idx_var, parameter_to_vary)
                cfg = config.assign_default_values(cfg)
                dft_props = config.set_stft_properties(cfg['stft'], cfg['fs'])
                is_cmwf_bf = cfg['cyclostationary_target']

                # Each iteration is a different montecarlo realization. Randomly vary: target signal, noise, ATF, etc.
                for idx_mtc in range(cfg['num_montecarlo_simulations']):
                    dg = data_generator.DataGenerator(cfg['target']['harmonic_correlation'],
                                                      cfg['noise']['harmonic_correlation'],
                                                      mean_random_proc=0.5 if is_cmwf_bf else 0.,
                                                      datasets_path=cfg['datasets_path'])
                    f0man = f0_manager.F0Manager()

                    if dg.sin_gen['target'].mean_random_process == 0 and is_cmwf_bf and \
                            cfg['target']['sig_type'] == 'sinusoidal':
                        raise ValueError("Only tested with cMVDR, could have unintended consequences with cMWF")

                    do_plots = idx_mtc == 0 and (idx_var == 0 or idx_var == len(varying_param_values) - 1 or
                                                 plot_sett['all_variations'])
                    SFT, SFT_real, freqs_hz = dg.get_stft_objects(dft_props)
                    signals, max_len_ir_atf, target_ir = dg.generate_signals(cfg, SFT_real, dft_props)

                    harmonic_freqs_est, crb_dict, f0_over_time = f0man.estimate_f0_or_resonant_freqs(
                        signals, cfg, dft_props, sin_generators=dg.sin_gen,
                        do_plots=do_plots and cfg['plot']['f0_spectrogram'])

                    # Covariance estimation & beamforming
                    bfd_all_chunks_stft = ExperimentManager.run_cov_estimation_beamforming(
                        signals, f0man, f0_over_time, harmonic_freqs_est, cfg, dft_props, do_plots, SFT)

                    signals_bfd_dict = ExperimentManager.convert_signals_time_domain(bfd_all_chunks_stft, SFT_real)

                    if post_filtering:
                        signals_bfd_dict_backup = dcopy(signals_bfd_dict)
                        ExperimentManager.apply_post_filtering(signals, signals_bfd_dict, dft_props, f0man,
                                                               f0_over_time,
                                                               harmonic_freqs_est, do_plots, SFT, SFT_real,
                                                               cfg_original)

                    signals_dict = {**signals, **signals_bfd_dict_backup, **signals_bfd_dict}
                    signals_dict = evaluator.bake_dict_for_evaluation(signals_dict,
                                                                      needs_masked_stft=cfg[
                                                                          'use_masked_stft_for_evaluation'])

                    # Evaluate performance of beamformers. Single montecarlo and single parameter value (e.g., SNR = 0dB)
                    metrics_list_time = evaluator.adjust_metrics_list(cfg['metrics']['time'],
                                                                      config.ConfigManager.is_speech(cfg['target']))
                    results_dict[str(param_value)].append((
                        evaluator.evaluate_signals(signals_dict, metrics_list_time, cfg['metrics']['freq'], fs,
                                                   cfg['use_masked_stft_for_evaluation'],
                                                   reference_sig_name='wet_rank1',
                                                   K_nfft_real=dft_props['nfft_real'],
                                                   print_results=cfg['print_results'])))

                    # Store audio signals for all param_values and montecarlo realizations to listen to them later
                    signals_dict_all_variations_time[str(param_value)][idx_mtc] = {key: dcopy(signals_dict[key]['time'])
                                                                                   for
                                                                                   key in
                                                                                   signals_dict.keys()}

                    if 'oracle' not in cfg['harmonics_est']['algo'] and cfg['metrics']['other']:
                        harmonic_freqs_oracle = dg.sin_gen['noise'].freqs_synthetic_signal
                        res1 = evaluator.evaluate_frequency_estimation(harmonic_freqs_oracle, harmonic_freqs_est)
                        res2 = evaluator.evaluate_crb(crb_dict, harmonic_freqs_oracle, fs)
                        results_freq_est[str(param_value)].append({'freq-err-mae': res1 | res2})

                # end of montecarlo simulations loop
            # end of parameter variations loop

            plots_args_default = {'varying_param_values': varying_param_values, 'parameter_to_vary': parameter_to_vary}
            plot_args_bf, plot_args_freq_est = evaluator.rearrange_and_average_results_all(results_dict,
                                                                                           plots_args_default,
                                                                                           results_freq_est, 'Noisy')
            evaluator.log_intermediate_results(plot_args_bf, plot_args_freq_est, varying_param_values, plot_sett)
            results_data_type_plots[parameter_to_vary] = dcopy(plot_args_bf)
            results_data_type_freq_est_plots[parameter_to_vary] = dcopy(plot_args_freq_est)

            try:
                Player.play_signals(signals_dict_all_variations_time, fs)
            except Exception as exc:
                print(f"Could not play the signals: {exc}")

            print(f"Varying parameter: {parameter_to_vary}")

        # Path of this module
        module_path = Path(__file__).parent.parent
        target_path_figs = module_path / Path('figs') / datetime.now().strftime("%Y-%m-%d") / time.strftime(
            '%Hh%M')

        # Plot beamforming errors
        pl.visualize_all_results(results_data_type_plots, plot_sett, cfg_original, False, False,
                                 target_path_figs, True)

        # Plot errors for frequency estimation
        pl.visualize_all_results(results_data_type_freq_est_plots, plot_sett, cfg_original, plot_db=False,
                                 target_path_figs_=target_path_figs)
        pl.visualize_all_results(results_data_type_freq_est_plots, plot_sett, cfg_original, plot_db=True,
                                 target_path_figs_=target_path_figs)

        # Move *.pkl files from target_path_figs to target_path_figs / 'figs_pkl'
        if target_path_figs.exists() and any(target_path_figs.iterdir()):
            target_path_figs_pkl = target_path_figs / 'figs_pkl'
            target_path_figs_pkl.mkdir(parents=True, exist_ok=True)
            for pkl_file in target_path_figs.glob('*.pkl'):
                pkl_file.rename(target_path_figs_pkl / pkl_file.name)

        ret = {
            'results_data_type_plots': results_data_type_plots,
            'results_data_type_freq_est_plots': results_data_type_freq_est_plots,
            'signals_dict_all_variations_time': signals_dict_all_variations_time,
            'target_path_figs': target_path_figs,
            'cfg_original': cfg_original,
        }

        return ret

    def run_cmvdr_inference_folder(self, input_folder, output_folder, cfg):
        """
        Run cMVDR inference on a given dataset (folder) and save the beamformed output to a specified output folder.
        This method loads audio files from the input folder, applies cMVDR beamforming, and
        saves the results to the output folder.
        :param input_folder: Path to the folder containing input audio files.
        :param output_folder: Path to the folder where output audio files will be saved.
        :param cfg: Configuration dictionary containing parameters for the experiment.
        """

        dft_props = config.set_stft_properties(cfg['stft'], cfg['fs'])
        dg = DataGenerator()
        SFT, SFT_real, freqs_hz = dg.get_stft_objects(dft_props)

        audio_list, names = audio_loader.AudioDiskLoader().load_audio_files(input_folder)
        audio_list_stft = [SFT_real.stft(x) for x in audio_list]

        # results_dict = {}
        # reference_sig_name = 'noisy'
        signals_dict_all_variations_time = {}
        f0man = f0_manager.F0Manager()

        for waveform, audio_stft, name in tqdm(
                zip(audio_list, audio_list_stft, names),
                total=len(audio_list)
        ):

            signals = {'noisy': {'stft': np.asarray(audio_stft)[np.newaxis], 'time': np.asarray(waveform)[np.newaxis]}}
            signals['noisy']['stft_conj'] = np.conj(signals['noisy']['stft'])

            # Estimate resonant frequencies from the noisy signal
            harmonic_freqs_est, crb_dict, f0_over_time = f0man.estimate_f0_or_resonant_freqs(signals, cfg, dft_props)

            # print("Debug: using noisy signal as noise_cov_est for estimating noise covariance.")
            signals['noise_cov_est'] = signals['noisy']

            bfd_stft_inference = self.run_cov_estimation_beamforming(signals=signals, f0man=f0man,
                                                                     f0_over_time=f0_over_time,
                                                                     harmonic_freqs_est=harmonic_freqs_est, cfg=cfg,
                                                                     dft_props=dft_props, SFT=SFT,
                                                                     name_input_sig='noisy')

            signals_bfd_dict = ExperimentManager.convert_signals_time_domain(bfd_stft_inference, SFT_real)
            signals = {**signals, **signals_bfd_dict}
            signals = evaluator.bake_dict_for_evaluation(signals)

            # Store audio signals for all param_values and montecarlo realizations to listen to them later
            signals_dict_all_variations_time[name] = {key: dcopy(signals[key]['time']) for key in signals.keys()}

        audio_loader.AudioDiskLoader.save_audio_files(
            signals_dict_all_variations_time, output_folder, fs=cfg['fs'], export_list=['cmvdr_blind'])
