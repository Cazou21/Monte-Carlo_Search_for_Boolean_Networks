#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=str, default="../results")
    p.add_argument("--figdir", type=str, default="../results/figs_compare_fastpairs")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--nb_chunk", type=int, required=True)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--ensemble_size", type=int, default=None)
    p.add_argument("--sims_per_model", type=int, default=None)
    p.add_argument("--grid_points", type=int, default=1000)
    return p.parse_args()


def normalize_value(x):
    if pd.isna(x):
        return "none"
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf)
    except Exception:
        return str(x)


def filter_history_for_plot(df, algo):
    if df.empty:
        return df

    if "event_type" in df.columns:
        if algo == "BILNMCS":
            if "evaluator" not in df.columns:
                raise ValueError("Column 'evaluator' missing for BILNMCS history.")
            df = df[
                (df["evaluator"] == "main") &
                (df["event_type"] == "improvement")
            ].copy()
        else:
            df = df[df["event_type"] == "improvement"].copy()
    else:
        # Compatibilité ancienne version
        if algo == "BILNMCS":
            if "cache_prefix" not in df.columns:
                raise ValueError("Neither event_type/evaluator nor cache_prefix found.")
            df = df[df["cache_prefix"] == "main"].copy()

    return df


def load_algo_history(indir, exp_name, algo, nb_chunk):
    pattern = os.path.join(
        indir,
        f"julia_algo_history_exp_{exp_name}_algo_{algo}_depth_*_timeout_*_ens_*_sims_*_fastens_*_fastsims_*_trial_*_chunk_*_of_{nb_chunk}.csv"
    )
    files = sorted(glob.glob(pattern))

    print(f"[DEBUG] {algo} pattern: {pattern}")
    print(f"[DEBUG] {algo} files found: {len(files)}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def build_step_curve(trial_df, timeout, grid):
    if trial_df.empty:
        return None

    trial_df = trial_df.sort_values("time_sec").reset_index(drop=True)

    times = trial_df["time_sec"].to_numpy(dtype=float)
    scores = trial_df["score"].to_numpy(dtype=float)

    valid = np.isfinite(times) & np.isfinite(scores)
    times = times[valid]
    scores = scores[valid]

    if len(times) == 0:
        return None

    keep = (times >= 0) & (times <= timeout)
    times = times[keep]
    scores = scores[keep]

    if len(times) == 0:
        return None

    y = np.full_like(grid, np.nan, dtype=float)

    first_time = times[0]
    first_score = scores[0]

    y[grid < max(first_time, 1e-12)] = first_score

    idx = np.searchsorted(times, grid, side="right") - 1
    mask = idx >= 0
    y[mask] = scores[idx[mask]]

    y[grid > times[-1]] = scores[-1]

    return y


def collect_trial_curves(df, algo, timeout, grid):
    curves = []
    trial_groups = df.groupby(["trial", "chunk_id"], dropna=False)

    for (trial, chunk_id), trial_df in trial_groups:
        try:
            trial_df = filter_history_for_plot(trial_df, algo)
        except Exception as e:
            print(f"[WARN] skip trial={trial}, chunk={chunk_id} for {algo}: {e}")
            continue

        if trial_df.empty:
            continue

        curve = build_step_curve(trial_df, timeout, grid)
        if curve is None:
            continue

        curves.append(curve)

    if not curves:
        return None

    return np.vstack(curves)


def build_log2_ticks(timeout):
    ticks = []
    v = 1
    while v <= timeout:
        ticks.append(v)
        v *= 2

    if ticks[-1] != timeout and timeout > ticks[-1]:
        ticks.append(timeout)

    labels = []
    for t in ticks:
        if abs(t - round(t)) < 1e-9:
            labels.append(str(int(round(t))))
        else:
            labels.append(str(t))

    return ticks, labels


def main():
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    lnmcs_df = load_algo_history(args.indir, args.exp_name, "LNMCS", args.nb_chunk)
    bilnmcs_df = load_algo_history(args.indir, args.exp_name, "BILNMCS", args.nb_chunk)

    if lnmcs_df.empty:
        print("[WARN] no LNMCS files found")
        return

    if bilnmcs_df.empty:
        print("[WARN] no BILNMCS files found")
        return

    required_cols = [
        "depth",
        "timeout",
        "trial",
        "chunk_id",
        "ensemble_size",
        "sims_per_model",
        "ensemble_size_fast",
        "sims_per_model_fast",
        "time_sec",
        "score",
    ]

    for name, df in [("LNMCS", lnmcs_df), ("BILNMCS", bilnmcs_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[WARN] missing columns for {name}: {missing}")
            return

    lnmcs_df = lnmcs_df[lnmcs_df["timeout"] == args.timeout].copy()
    bilnmcs_df = bilnmcs_df[bilnmcs_df["timeout"] == args.timeout].copy()

    if args.depth is not None:
        lnmcs_df = lnmcs_df[lnmcs_df["depth"] == args.depth].copy()
        bilnmcs_df = bilnmcs_df[bilnmcs_df["depth"] == args.depth].copy()

    if args.ensemble_size is not None:
        lnmcs_df = lnmcs_df[lnmcs_df["ensemble_size"] == args.ensemble_size].copy()
        bilnmcs_df = bilnmcs_df[bilnmcs_df["ensemble_size"] == args.ensemble_size].copy()

    if args.sims_per_model is not None:
        lnmcs_df = lnmcs_df[lnmcs_df["sims_per_model"] == args.sims_per_model].copy()
        bilnmcs_df = bilnmcs_df[bilnmcs_df["sims_per_model"] == args.sims_per_model].copy()

    if lnmcs_df.empty or bilnmcs_df.empty:
        print("[WARN] empty dataframe after filtering")
        return

    fast_pairs = sorted(
        set(
            zip(
                bilnmcs_df["ensemble_size_fast"].tolist(),
                bilnmcs_df["sims_per_model_fast"].tolist()
            )
        )
    )

    grid = np.geomspace(1, args.timeout, args.grid_points)
    ticks, tick_labels = build_log2_ticks(args.timeout)

    for fast_ens, fast_sims in fast_pairs:
        bil_sub = bilnmcs_df[
            (bilnmcs_df["ensemble_size_fast"] == fast_ens) &
            (bilnmcs_df["sims_per_model_fast"] == fast_sims)
        ].copy()

        combos = bil_sub[["depth", "ensemble_size", "sims_per_model"]].drop_duplicates()

        for _, combo in combos.iterrows():
            depth = combo["depth"]
            ens = combo["ensemble_size"]
            sims = combo["sims_per_model"]

            bil_cfg = bil_sub[
                (bil_sub["depth"] == depth) &
                (bil_sub["ensemble_size"] == ens) &
                (bil_sub["sims_per_model"] == sims)
            ].copy()

            lnmcs_cfg = lnmcs_df[
                (lnmcs_df["depth"] == depth) &
                (lnmcs_df["ensemble_size"] == ens) &
                (lnmcs_df["sims_per_model"] == sims)
            ].copy()

            if lnmcs_cfg.empty or bil_cfg.empty:
                print(
                    f"[WARN] skip depth={depth}, ens={ens}, sims={sims}, "
                    f"fastens={fast_ens}, fastsims={fast_sims} because one side is empty"
                )
                continue

            lnmcs_curves = collect_trial_curves(lnmcs_cfg, "LNMCS", args.timeout, grid)
            bilnmcs_curves = collect_trial_curves(bil_cfg, "BILNMCS", args.timeout, grid)

            if lnmcs_curves is None or bilnmcs_curves is None:
                print(
                    f"[WARN] no plottable curves for depth={depth}, ens={ens}, sims={sims}, "
                    f"fastens={fast_ens}, fastsims={fast_sims}"
                )
                continue

            lnmcs_min = np.nanmin(lnmcs_curves, axis=0)
            lnmcs_mean = np.nanmean(lnmcs_curves, axis=0)
            lnmcs_max = np.nanmax(lnmcs_curves, axis=0)

            bil_min = np.nanmin(bilnmcs_curves, axis=0)
            bil_mean = np.nanmean(bilnmcs_curves, axis=0)
            bil_max = np.nanmax(bilnmcs_curves, axis=0)

            plt.figure(figsize=(10, 6))

            plt.fill_between(grid, bil_min, bil_max, alpha=0.25, label="BILNMCS min-max")
            plt.plot(grid, bil_mean, label="mean BILNMCS")

            plt.plot(grid, lnmcs_mean, color="orange", label="mean LNMCS")
            plt.plot(grid, lnmcs_min, color="orange", linestyle="--", label="min LNMCS")
            plt.plot(grid, lnmcs_max, color="orange", linestyle="--", label="max LNMCS")

            plt.xscale("log", base=2)
            plt.xlim(1, args.timeout)
            plt.xticks(ticks, tick_labels)

            plt.xlabel("Temps (s, échelle log2)")
            plt.ylabel("Best score so far")
            plt.title(
                f"exp={args.exp_name} | depth={depth} | timeout={args.timeout} | "
                f"ens={ens} | sims={sims} | fastens={normalize_value(fast_ens)} | "
                f"fastsims={normalize_value(fast_sims)}"
            )
            plt.grid(True, which="both")
            plt.legend()

            out_png = os.path.join(
                args.figdir,
                f"compare_meanLNMCS_vs_BILNMCSbands_"
                f"{args.exp_name}_depth_{depth}_timeout_{normalize_value(args.timeout)}_"
                f"ens_{normalize_value(ens)}_sims_{normalize_value(sims)}_"
                f"fastens_{normalize_value(fast_ens)}_fastsims_{normalize_value(fast_sims)}.png"
            )

            plt.savefig(out_png, bbox_inches="tight")
            plt.close()

            print(f"saved: {out_png}")


if __name__ == "__main__":
    main()