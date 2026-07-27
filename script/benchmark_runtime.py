import argparse
import copy
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from cmvdr.beamforming.beamformer_manager import BeamformerManager
from cmvdr.data_gen.data_generator import DataGenerator
from cmvdr.data_gen.f0_manager import F0ChangeAmount, F0Manager
from cmvdr.data_gen.manager import Manager
from cmvdr.estimation.covariance_estimator import CovarianceEstimator
from cmvdr.estimation.modulator import Modulator
from cmvdr.util import config
from cmvdr.util.harmonic_info import HarmonicInfo
from cmvdr.util import globs as gs


DEFAULT_DATASET_CONFIGS = {
    "Synthetic": "benchmarks/benchmark_synthetic.yaml",
    "MUSAN": "benchmarks/benchmark_freesound.yaml",
    "DREGON": "benchmarks/benchmark_dregon.yaml",
}


@dataclass
class StageTimings:
    cyclic_s: float = 0.0
    covariance_s: float = 0.0
    beamformer_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.cyclic_s + self.covariance_s + self.beamformer_s


OPTIONAL_SUMMARY_FIELDS = ("avg_ck", "coherent_bins_pct")


def _set_seed(cfg):
    gs.rng, cfg["seed_extracted"] = gs.compute_rng(cfg["seed_is_random"], cfg["seed_if_not_random"])


def _set_printoptions_numpy():
    """ Set numpy print options to make it easier to read. Also set pprint as default for dict() """
    desired_width = 180  # 220
    np.set_printoptions(precision=2, linewidth=desired_width, suppress=True)


def _resolve_config_path(name_or_path):
    path = Path(name_or_path)
    if path.exists():
        return path

    project_root = config.get_project_root()
    candidates = [
        project_root / "configs" / name_or_path,
        project_root / "configs" / "benchmarks" / name_or_path,
        project_root / "configs" / "experiments" / name_or_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find configuration file: {name_or_path}")


def _load_config_with_base(name_or_path):
    path = _resolve_config_path(name_or_path)
    cfg = config.load_yaml_from_path(path)
    base = cfg.get("base") or cfg.get("inherits") or None
    if base:
        base_cfg = _load_config_with_base(base)
        cfg = config.merge_configurations(base_cfg, cfg)
    return cfg


def _prepare_cfg(cfg):
    cfg = config.assign_default_values(cfg)
    cfg["target"] = config.update_target_settings(cfg["target"])
    config.check_cyclic_target_or_not(cfg)
    cfg["beamforming"]["methods"] = BeamformerManager.infer_beamforming_methods(cfg["beamforming"])
    return cfg


def _benchmark_dataset_cfg(benchmark_cfg, dataset_cfg_name):
    dataset_cfg = _load_config_with_base(dataset_cfg_name)
    merged = config.merge_configurations(benchmark_cfg, dataset_cfg)
    merged = _prepare_cfg(merged)
    _set_seed(merged)
    return merged


def _make_stft(cfg):
    dft_props = config.set_stft_properties(copy.deepcopy(cfg["stft"]), cfg["fs"])
    dg = DataGenerator(
        cfg["target"]["harmonic_correlation"],
        cfg["noise"]["harmonic_correlation"],
        mean_random_proc=0.5 if cfg["cyclostationary_target"] else 0.0,
        datasets_path=cfg["datasets_path"],
    )
    SFT, SFT_real, _ = dg.get_stft_objects(dft_props)
    return dg, SFT, SFT_real, dft_props


def _prepare_mpdr_signals(signals, SFT):
    # MPDR still needs the reshaped signal layout for covariance estimation and beamforming.
    return Modulator.modulate_signals(signals, ["noisy"], SFT, [np.array([0])], P_max=1, name_input_sig="noisy")


def _coherence_metrics(harm_info):
    coherent_mask = np.asarray(getattr(harm_info, "mask_harmonic_bins", np.array([])), dtype=bool)
    if coherent_mask.size == 0:
        return {"avg_ck": None, "coherent_bins_pct": 0.0}

    total_bins = int(coherent_mask.size)
    coherent_bins = int(np.count_nonzero(coherent_mask))
    if coherent_bins == 0:
        return {"avg_ck": None, "coherent_bins_pct": 0.0}

    ck_all = np.asarray(harm_info.get_num_shifts_all_frequencies(), dtype=float)
    avg_ck = float(np.mean(ck_all[coherent_mask])) if ck_all.size else None
    coherent_bins_pct = 100.0 * coherent_bins / total_bins
    return {"avg_ck": avg_ck, "coherent_bins_pct": coherent_bins_pct}


def _timed_sample(cfg, algorithm, warmup=False):
    dg, SFT, SFT_real, dft_props = _make_stft(cfg)
    f0man = F0Manager()
    signals, _, _ = dg.generate_signals(cfg, SFT_real, dft_props)

    coherence_metrics = {"avg_ck": None, "coherent_bins_pct": 0.0}
    cyclic_timings = 0.0
    if algorithm == "mpdr":
        signals = _prepare_mpdr_signals(signals, SFT)
        harmonic_freqs_est = np.array([])
    else:
        t0 = perf_counter()
        harmonic_freqs_est, _, f0_over_time = f0man.estimate_f0_or_resonant_freqs(
            signals, cfg, dft_props, sin_generators=None, do_plots=False
        )
        cyclic_timings = perf_counter() - t0

    m = Manager()
    cov_est = CovarianceEstimator(
        cfg["cov_estimation"],
        cfg["cyclostationary_target"],
        subtract_mean=False,
        use_pseudo_cov=any("wl" in name for name in cfg["beamforming"]["methods"]),
    )
    cov_est.harmonic_info = HarmonicInfo()
    bf = BeamformerManager(
        beamformers_names=cfg["beamforming"]["methods"],
        sig_shape_k_m=(dft_props["nfft_real"], signals["noisy"]["stft"].shape[0]),
        minimize_noisy_cov_mvdr=cfg["beamforming"]["minimize_noisy_cov_mvdr"],
        loadings=cfg["beamforming"]["loadings"],
        noise_var_rtf=cfg["noise"]["noise_var_rtf"],
    )
    bf.harmonic_info = HarmonicInfo()

    slice_bf_list, slice_cov_est_list, num_chunks = m.get_chunks_slices(
        signals["noisy"]["stft"].shape[-1], dft_props=dft_props, time_props=cfg["time"],
        recursive_average=cfg["cov_estimation"]["recursive_average"]
    )

    cov_dict_prev = {}
    timings = StageTimings(cyclic_s=cyclic_timings)

    for idx_chunk, (slice_bf, slice_cov_est) in enumerate(zip(slice_bf_list, slice_cov_est_list)):
        is_first_chunk = idx_chunk == 0
        mod_amount = F0ChangeAmount.no_change
        harm_info = HarmonicInfo()

        if algorithm == "cmpdr":
            t0 = perf_counter()
            harmonic_freqs_chunk = harmonic_freqs_est

            if is_first_chunk:
                harm_info, mod_amount = f0man.compute_harmonic_and_modulation_sets_global_coherence(
                        signals[cfg["cyclic"]["coherence_source_signal_name"]],
                        harmonic_freqs_chunk,
                        SFT,
                        cfg["cyclic"],
                    )
                coherence_metrics = _coherence_metrics(harm_info)

            signals_to_modulate = config.ConfigManager.choose_signals_to_modulate(
                cfg["cyclostationary_target"],
                cfg["beamforming"]["minimize_noisy_cov_mvdr"],
                is_first_chunk,
                cfg["cov_estimation"]["recursive_average"],
                "noisy",
                skip_noise_cov_est=cfg["data_type"] == "inference",
            )
            if mod_amount == F0ChangeAmount.small:
                signals = Modulator.modulate_signals(
                    signals,
                    signals_to_modulate,
                    SFT,
                    harm_info.alpha_mods_sets,
                    cfg["cyclic"]["P_max"],
                    "noisy",
                )
            bf.harmonic_info = harm_info
            cov_est.harmonic_info = harm_info
            timings.cyclic_s += perf_counter() - t0

        cov_est.set_dimensions((dft_props["nfft_real"], signals["noisy"]["stft"].shape[0], cfg["cyclic"]["P_max"]))
        t0 = perf_counter()
        cov_dict = cov_est.estimate_covariances(
            slice_cov_est,
            signals,
            cov_dict_prev,
            num_mics_changed=cov_est.sig_shape_k_m_p[1] != signals["noisy"]["stft"].shape[0],
            modulation_amount=mod_amount,
            name_input_sig="noisy",
        )
        timings.covariance_s += perf_counter() - t0
        cov_dict_prev = copy.deepcopy(cov_dict)

        target_rtf = np.array([])
        if 'wet' in signals:
            target_rtf = DataGenerator.calculate_ground_truth_rtf(signals['wet'])

        t0 = perf_counter()
        weights, _ = bf.compute_weights_all_beamformers(cov_dict=cov_dict, idx_chunk=idx_chunk, name_input_sig="noisy",
                                                        rtf_oracle=target_rtf)
        bf.beamform_signals(
            signals["noisy"]["stft"],
            signals["noisy"]["mod_stft_3d"],
            slice_bf,
            weights,
            mod_amount=mod_amount,
        )
        timings.beamformer_s += perf_counter() - t0

    if warmup:
        return None

    sample_label = Path(str(cfg.get("noise", {}).get("sample_path", ""))).name or Path(
        str(cfg["target"].get("sample_name", ""))
    ).name
    return {
        "sample_label": sample_label,
        "cyclic_s": timings.cyclic_s,
        "covariance_s": timings.covariance_s,
        "beamformer_s": timings.beamformer_s,
        "total_s": timings.total_s,
        "num_chunks": num_chunks,
        "avg_ck": coherence_metrics["avg_ck"],
        "coherent_bins_pct": coherence_metrics["coherent_bins_pct"],
    }


def _mean_or_none(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    summary_rows = []
    overall = {"dataset": "Overall", "runtime_mean_s": 0.0, "samples": 0}
    total_runtime = 0.0
    total_n = 0

    for dataset in sorted(grouped):
        dataset_rows = grouped[dataset]
        runtimes = [r["runtime_s"] for r in dataset_rows]
        n = len(dataset_rows)
        row = {
            "dataset": dataset,
            "algorithm": dataset_rows[0]["algorithm"],
            "runtime_mean_s": float(np.mean(runtimes)),
            "samples": n,
        }
        for field in OPTIONAL_SUMMARY_FIELDS:
            mean_value = _mean_or_none([r.get(field) for r in dataset_rows])
            if mean_value is not None:
                row[f"{field}_mean"] = mean_value
        summary_rows.append(row)
        total_runtime += float(np.mean(runtimes)) * n
        total_n += n

    if total_n:
        overall["runtime_mean_s"] = total_runtime / total_n
        overall["samples"] = total_n
        for field in OPTIONAL_SUMMARY_FIELDS:
            values = [r.get(f"{field}_mean", r.get(field)) for r in summary_rows]
            mean_value = _mean_or_none(values)
            if mean_value is not None:
                overall[f"{field}_mean"] = mean_value
        summary_rows.append(overall)

    return summary_rows


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary_rows, title):
    print(title)
    if summary_rows and "runtime_mean_s" in summary_rows[0] and "mpdr_mean_s" not in summary_rows[0]:
        has_coherence = "avg_ck_mean" in summary_rows[0] or "coherent_bins_pct_mean" in summary_rows[0]
        header = "Dataset\tRuntime"
        if has_coherence:
            header += "\tAvg Ck\tCoherent bins %"
        header += "\tN"
        print(header)
        for row in summary_rows:
            line = f"{row['dataset']}\t{row['runtime_mean_s']:.4f}"
            if has_coherence:
                avg_ck = row.get("avg_ck_mean")
                coherent_bins_pct = row.get("coherent_bins_pct_mean")
                line += f"\t{avg_ck:.4f}" if avg_ck is not None else "\t-"
                line += f"\t{coherent_bins_pct:.2f}%" if coherent_bins_pct is not None else "\t-"
            line += f"\t{row['samples']}"
            print(line)
    else:
        print("Dataset\tMPDR\tcMPDR\tSlowdown\tAvg Ck\tCoherent bins %\tN")
        for row in summary_rows:
            avg_ck = row.get("avg_ck_mean")
            coherent_bins_pct = row.get("coherent_bins_pct_mean")
            avg_ck_text = f"{avg_ck:.4f}" if avg_ck is not None else "-"
            pct_text = f"{coherent_bins_pct:.2f}%" if coherent_bins_pct is not None else "-"
            print(
                f"{row['dataset']}\t{row['mpdr_mean_s']:.4f}\t{row['cmpdr_mean_s']:.4f}\t"
                f"{row['slowdown_x']:.2f}x\t{avg_ck_text}\t{pct_text}\t{row['samples']}"
            )


def run_benchmark(benchmark_cfg_path):
    benchmark_cfg = _load_config_with_base(benchmark_cfg_path)
    benchmark_section = benchmark_cfg.get("benchmark", {})
    dataset_cfgs = benchmark_section.get("datasets", DEFAULT_DATASET_CONFIGS)
    algorithm = benchmark_section.get("algorithm")
    if algorithm not in {"mpdr", "cmpdr"}:
        raise ValueError("benchmark.algorithm must be 'mpdr' or 'cmpdr'.")

    sample_limit = int(benchmark_section.get("sample_limit", benchmark_cfg.get("num_montecarlo_simulations", 10)))
    warmup_runs = int(benchmark_section.get("warmup_runs", 1))
    output_root = Path(benchmark_section.get("output_dir", "./exp_results/benchmarks")).expanduser().resolve()
    run_stamp = datetime.now().strftime("%Y-%m-%d/%Hh%M_%S")
    run_root = output_root / run_stamp / Path(str(benchmark_cfg_path)).stem

    raw_rows = []
    for dataset_name, dataset_cfg_path in dataset_cfgs.items():
        print(f"Benchmarking {algorithm.upper()} on dataset '{dataset_name}' with {sample_limit} samples...")
        dataset_cfg = _benchmark_dataset_cfg(benchmark_cfg, dataset_cfg_path)
        dataset_cfg["num_montecarlo_simulations"] = sample_limit
        dataset_cfg["benchmark_dataset"] = dataset_name

        for _ in range(warmup_runs):
            _timed_sample(copy.deepcopy(dataset_cfg), algorithm, warmup=True)

        for idx in range(sample_limit):
            sample_result = _timed_sample(copy.deepcopy(dataset_cfg), algorithm)
            raw_rows.append(
                {
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                    "sample_idx": idx,
                    "sample_label": sample_result["sample_label"],
                    "runtime_s": sample_result["total_s"],
                    "cyclic_s": sample_result["cyclic_s"],
                    "covariance_s": sample_result["covariance_s"],
                    "beamformer_s": sample_result["beamformer_s"],
                    "num_chunks": sample_result["num_chunks"],
                    "avg_ck": sample_result["avg_ck"],
                    "coherent_bins_pct": sample_result["coherent_bins_pct"],
                }
            )

    summary_rows = _aggregate_rows(raw_rows)

    raw_path = run_root / f"{algorithm}_timings.jsonl"
    summary_path = run_root / f"{algorithm}_summary.csv"
    _write_jsonl(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _print_summary(summary_rows, f"{algorithm.upper()} benchmark")
    print(f"Raw timings: {raw_path}")
    print(f"Summary: {summary_path}")
    return summary_path


def _load_summary_csv(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def compare_summaries(mpdr_summary_path, cmpdr_summary_path, output_path=None):
    mpdr_rows = _load_summary_csv(mpdr_summary_path)
    cmpdr_rows = _load_summary_csv(cmpdr_summary_path)

    mpdr_map = {row["dataset"]: row for row in mpdr_rows if row["dataset"] != "Overall"}
    cmpdr_map = {row["dataset"]: row for row in cmpdr_rows if row["dataset"] != "Overall"}
    datasets = sorted(set(mpdr_map) & set(cmpdr_map))

    rows = []
    total_mpdr = 0.0
    total_cmpdr = 0.0
    total_avg_ck = 0.0
    total_pct = 0.0
    total_n = 0
    have_avg_ck = False
    have_pct = False
    for dataset in datasets:
        mpdr = mpdr_map[dataset]
        cmpdr = cmpdr_map[dataset]
        n = int(float(mpdr["samples"]))
        mpdr_mean = float(mpdr["runtime_mean_s"])
        cmpdr_mean = float(cmpdr["runtime_mean_s"])
        slowdown = float(cmpdr_mean / mpdr_mean) if mpdr_mean > 0 else float("nan")
        avg_ck = cmpdr.get("avg_ck_mean")
        avg_ck = float(avg_ck) if avg_ck not in (None, "") else None
        pct = cmpdr.get("coherent_bins_pct_mean")
        pct = float(pct) if pct not in (None, "") else None
        rows.append(
            {
                "dataset": dataset,
                "mpdr_mean_s": mpdr_mean,
                "cmpdr_mean_s": cmpdr_mean,
                "slowdown_x": slowdown,
                "avg_ck_mean": avg_ck,
                "coherent_bins_pct_mean": pct,
                "samples": n,
            }
        )
        total_mpdr += mpdr_mean * n
        total_cmpdr += cmpdr_mean * n
        if avg_ck is not None:
            total_avg_ck += avg_ck * n
            have_avg_ck = True
        if pct is not None:
            total_pct += pct * n
            have_pct = True
        total_n += n

    if total_n:
        overall_avg_ck = (total_avg_ck / total_n) if have_avg_ck else None
        overall_pct = (total_pct / total_n) if have_pct else None
        rows.append(
            {
                "dataset": "Overall",
                "mpdr_mean_s": total_mpdr / total_n,
                "cmpdr_mean_s": total_cmpdr / total_n,
                "slowdown_x": (total_cmpdr / total_n) / (total_mpdr / total_n),
                "avg_ck_mean": overall_avg_ck,
                "coherent_bins_pct_mean": overall_pct,
                "samples": total_n,
            }
        )

    _print_summary(rows, "Average runtime")
    if output_path is None:
        output_path = Path("benchmark_runtime_table.csv")
    output_path = Path(output_path)
    _write_csv(output_path, rows)
    print(f"Comparison table: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark MPDR and cMPDR runtime.")
    parser.add_argument("-c", "--config", help="Benchmark config file, e.g. experiments/benchmark_mpdr.yaml")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MPDR_SUMMARY", "CMPDR_SUMMARY"),
        help="Combine two benchmark summary CSV files into a comparison table.",
    )
    parser.add_argument("-o", "--output", help="Output CSV path for comparison mode.")
    args = parser.parse_args()

    if args.compare:
        compare_summaries(args.compare[0], args.compare[1], args.output)
        return

    if not args.config:
        raise SystemExit("Either --config or --compare must be provided.")

    _set_printoptions_numpy()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
