#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
import random
import hashlib
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm import tqdm

import bn_async_sim as bnas
import ensemble_utils as eu

from nmcs_module import nmcs
from lnmcs_module import lnmcs
from nrpa_module import nrpa
from gnrpa_module import gnrpa
from bilnmcs_module import bilnmcs

print("~~ init Julia")
t0 = timer()
import jl_impl
print(f"-- init Julia done [{timer()-t0:.2f}s]")


def load_experiment_config(config_path, exp_name):
    with open(config_path) as fp:
        config = json.load(fp)

    cfg = config[exp_name]
    init = cfg["init"]
    if isinstance(init, str):
        with open(init) as fp:
            init = json.load(fp)

    return cfg, init


def normalize_move_pair(move):
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        raise ValueError(f"Unexpected move format: {move!r}")

    a, b = move
    if isinstance(a, str):
        return (a, bool(b))
    if isinstance(b, str):
        return (b, bool(a))
    raise ValueError(f"Cannot normalize move: {move!r}")


def build_all_moves(ens_dir, output_nodes):
    model_files = sorted(f for f in os.listdir(ens_dir) if f.endswith(".bnet"))
    if not model_files:
        raise RuntimeError(f"No .bnet files found in {ens_dir}")

    first_model_path = os.path.join(ens_dir, model_files[0])
    network = bnas.load_network_model(first_model_path)

    raw = eu.generate_single_mutants(network, output_nodes)
    raw_moves = [tuple(m[0]) if isinstance(m, list) else tuple(m) for m in raw]
    return [normalize_move_pair(m) for m in raw_moves]


def decode_single_move(move, all_moves):
    if isinstance(move, int):
        if move < 0 or move >= len(all_moves):
            raise ValueError(f"Move index out of range: {move} (len(all_moves)={len(all_moves)})")
        return normalize_move_pair(all_moves[move])

    while isinstance(move, list) and len(move) == 1:
        move = move[0]

    return normalize_move_pair(move)


def has_duplicate_genes(muts, all_moves):
    if muts is None:
        return False
    genes = [decode_single_move(move, all_moves)[0] for move in muts]
    return len(genes) != len(set(genes))


def moves_to_mutant_dict(muts, all_moves):
    if muts is None:
        return {}

    if has_duplicate_genes(muts, all_moves):
        decoded = [decode_single_move(move, all_moves) for move in muts]
        raise ValueError(f"Duplicate gene in mutation sequence: {decoded}")

    mutant = {}
    for move in muts:
        gene, value = decode_single_move(move, all_moves)
        mutant[gene] = int(bool(value))
    return mutant


def make_stable_seed(*parts):
    seed_input = "|".join(str(x) for x in parts)
    return int(hashlib.md5(seed_input.encode("utf-8")).hexdigest()[:8], 16)


def build_experiment(cfg, init, ensemble_size, sims_per_model, max_steps, sample_at_evaluate=False):
    t0 = timer()
    main_ens = jl_impl.BooleanNetworkEnsemble(
        cfg["ens_dir"],
        cfg["ens_name"],
        force_rebuild=False
    )
    print(f"-- loaded ensemble [{timer()-t0:.2f}s]")

    t0 = timer()
    exp = jl_impl.Experiment(
        main_ens,
        ensemble_size,
        sims_per_model,
        max_steps,
        init,
        cfg["target"],
        init_rand=cfg.get("init_rand"),
        sample_at_evaluate=sample_at_evaluate
    )
    print(f"-- prepare experiment [{timer()-t0:.2f}s]")
    print(f"-- actual_size={exp.actual_size}")
    print(f"-- sample_at_evaluate={sample_at_evaluate}")
    return exp


def make_tracked_eval_fn(
    exp,
    all_moves,
    cache,
    evaluator,
    trial_history,
    t_start,
    best_tracker,
    use_cache=True,
    reject_duplicates=True
):
    def evaluate_fn(muts):
        if reject_duplicates and has_duplicate_genes(muts, all_moves):
            return -1e18

        mutant_dict = moves_to_mutant_dict(muts, all_moves)
        key = (evaluator, tuple(sorted(mutant_dict.items())))

        if use_cache and key in cache:
            score = cache[key]
        else:
            score = exp.evaluate(mutant_dict)
            if use_cache:
                cache[key] = score

        elapsed = time.time() - t_start

        if score > best_tracker["value"]:
            best_tracker["value"] = score
            best_tracker["mutant_dict"] = mutant_dict
            trial_history.append({
                "time_sec": elapsed,
                "score": score,
                "evaluator": evaluator,
                "event_type": "improvement",
                "mutant_dict": str(mutant_dict)
            })

        return score

    return evaluate_fn


def run_one_algo(algo, depth, timeout, all_moves, eval_main_fn,
                 eval_fast_fn=None,
                 nmcs_level=2,
                 lnmcs_level=2,
                 nrpa_level=50,
                 gnrpa_level=2,
                 gnrpa_N=100,
                 gnrpa_tau=0.5,
                 bilnmcs_level=2,
                 bilnmcs_b=2,
                 bilnmcs_r=0.5,
                 bilnmcs_e=None,
                 lnmcs_b=3,
                 lnmcs_r=0.4,
                 lnmcs_e=None):

    if algo == "NMCS":
        score, muts, eval_count = nmcs(
            state=[],
            level=nmcs_level,
            depth=depth,
            best_moves=all_moves,
            evaluate_fn=eval_main_fn,
            timeout_sec=timeout
        )
        return {
            "score": score,
            "mutations": muts,
            "eval_count": eval_count,
            "playout_count": None,
            "main_eval_count": None,
            "fast_eval_count": None,
            "algo_level": nmcs_level,
            "algo_b": None,
            "algo_r": None,
            "algo_e": None,
            "algo_tau": None,
            "algo_N": None,
        }

    if algo == "LNMCS":
        score, muts, eval_count = lnmcs(
            state=[],
            level=lnmcs_level,
            depth=depth,
            all_moves=all_moves,
            evaluate_fn=eval_main_fn,
            b=lnmcs_b,
            r=lnmcs_r,
            e=lnmcs_e,
            timeout_sec=timeout
        )
        return {
            "score": score,
            "mutations": muts,
            "eval_count": eval_count,
            "playout_count": None,
            "main_eval_count": None,
            "fast_eval_count": None,
            "algo_level": lnmcs_level,
            "algo_b": lnmcs_b,
            "algo_r": lnmcs_r,
            "algo_e": lnmcs_e,
            "algo_tau": None,
            "algo_N": None,
        }

    if algo == "NRPA":
        score, muts, playout_count, eval_count = nrpa(
            level=nrpa_level,
            policy={},
            depth=depth,
            evaluate_fn=eval_main_fn,
            all_moves=all_moves,
            timeout_sec=timeout
        )
        return {
            "score": score,
            "mutations": muts,
            "eval_count": eval_count,
            "playout_count": playout_count,
            "main_eval_count": None,
            "fast_eval_count": None,
            "algo_level": nrpa_level,
            "algo_b": None,
            "algo_r": None,
            "algo_e": None,
            "algo_tau": None,
            "algo_N": None,
        }

    if algo == "GNRPA":
        score, muts, _, _, _, _, eval_count = gnrpa(
            level=gnrpa_level,
            policy={},
            bias={},
            tau=gnrpa_tau,
            depth=depth,
            evaluate_fn=eval_main_fn,
            all_moves=all_moves,
            N=gnrpa_N,
            timeout_sec=timeout
        )
        return {
            "score": score,
            "mutations": muts,
            "eval_count": eval_count,
            "playout_count": None,
            "main_eval_count": None,
            "fast_eval_count": None,
            "algo_level": gnrpa_level,
            "algo_b": None,
            "algo_r": None,
            "algo_e": None,
            "algo_tau": gnrpa_tau,
            "algo_N": gnrpa_N,
        }

    if algo == "BILNMCS":
        if eval_fast_fn is None:
            raise ValueError("BILNMCS requires a fast evaluator")
        score, muts, main_eval_count, fast_eval_count = bilnmcs(
            state=[],
            level=bilnmcs_level,
            depth=depth,
            all_moves=all_moves,
            evaluate_main_fn=eval_main_fn,
            evaluate_fast_fn=eval_fast_fn,
            b=bilnmcs_b,
            r=bilnmcs_r,
            e=bilnmcs_e,
            timeout_sec=timeout
        )
        return {
            "score": score,
            "mutations": muts,
            "eval_count": None,
            "playout_count": None,
            "main_eval_count": main_eval_count,
            "fast_eval_count": fast_eval_count,
            "algo_level": bilnmcs_level,
            "algo_b": bilnmcs_b,
            "algo_r": bilnmcs_r,
            "algo_e": bilnmcs_e,
            "algo_tau": None,
            "algo_N": None,
        }

    raise ValueError(f"Unknown algorithm: {algo}")


def resolve_fast_pairs(args):
    if args.ensemble_sizes_fast is not None or args.sims_per_models_fast is not None:
        if args.ensemble_sizes_fast is None or args.sims_per_models_fast is None:
            raise ValueError(
                "Pour tester plusieurs paires fast, il faut fournir --ensemble_sizes_fast ET --sims_per_models_fast"
            )
        if len(args.ensemble_sizes_fast) != len(args.sims_per_models_fast):
            raise ValueError(
                "--ensemble_sizes_fast et --sims_per_models_fast doivent avoir la meme longueur"
            )
        pairs = list(zip(args.ensemble_sizes_fast, args.sims_per_models_fast))
    else:
        ensemble_size_fast = args.ensemble_size_fast
        sims_per_model_fast = args.sims_per_model_fast

        if ensemble_size_fast is None:
            ensemble_size_fast = args.ensemble_size

        pairs = [(ensemble_size_fast, sims_per_model_fast)]

    for ens_fast, sims_fast in pairs:
        if ens_fast < 1:
            raise ValueError(f"ensemble_size_fast must be >= 1, got {ens_fast}")
        if sims_fast < 1:
            raise ValueError(f"sims_per_model_fast must be >= 1, got {sims_fast}")

    return pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="../data/config.json")
    p.add_argument("--exp_name", type=str, default="TumourInvasion-WT")

    p.add_argument("--ensemble_size", type=int, default=1000)
    p.add_argument("--sims_per_model", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)

    p.add_argument("--ensemble_size_fast", type=int, default=None)
    p.add_argument("--sims_per_model_fast", type=int, default=2)

    p.add_argument("--ensemble_sizes_fast", type=int, nargs="+", default=None)
    p.add_argument("--sims_per_models_fast", type=int, nargs="+", default=None)

    p.add_argument("--use_fast_eval", action="store_true")
    p.add_argument("--sample_main_at_evaluate", action="store_true")

    p.add_argument("--algos", type=str, nargs="+",
                   default=["NRPA"],
                   choices=["NMCS", "LNMCS", "NRPA", "GNRPA", "BILNMCS"])

    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--timeouts", type=int, nargs="+", default=[180])
    p.add_argument("--n_trials", type=int, default=5)

    p.add_argument("--nmcs_level", type=int, default=2)

    p.add_argument("--lnmcs_level", type=int, default=2)
    p.add_argument("--lnmcs_b", type=int, default=3)
    p.add_argument("--lnmcs_r", type=float, default=0.4)
    p.add_argument("--lnmcs_e", type=int, default=None)

    p.add_argument("--nrpa_level", type=int, default=50)

    p.add_argument("--gnrpa_level", type=int, default=2)
    p.add_argument("--gnrpa_N", type=int, default=100)
    p.add_argument("--gnrpa_tau", type=float, default=0.5)

    p.add_argument("--bilnmcs_level", type=int, default=2)
    p.add_argument("--bilnmcs_b", type=int, default=2)
    p.add_argument("--bilnmcs_r", type=float, default=0.5)
    p.add_argument("--bilnmcs_e", type=int, default=None)

    p.add_argument("--chunk_id", type=int, required=True)
    p.add_argument("--num_chunks", type=int, required=True)

    p.add_argument("--outdir", type=str, default="../results")
    return p.parse_args()


def make_algo_suffixes(args):
    algos_set = set(args.algos)

    extra_parts = [
        f"ens_{args.ensemble_size}",
        f"sims_{args.sims_per_model}",
        f"mainsample_{int(bool(args.sample_main_at_evaluate))}",
    ]

    if "LNMCS" in algos_set:
        extra_parts.append(f"lnmcsb_{args.lnmcs_b}")

    if "BILNMCS" in algos_set:
        extra_parts.append(f"bilb_{args.bilnmcs_b}")

    return "_".join(extra_parts)


def main():
    args = parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if not (0 <= args.chunk_id < args.num_chunks):
        raise ValueError(f"chunk_id must be in [0, {args.num_chunks - 1}]")

    if "BILNMCS" in args.algos:
        args.use_fast_eval = True

    fast_pairs = resolve_fast_pairs(args)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"-- preparing experiment {args.exp_name}")
    cfg, init = load_experiment_config(args.config_path, args.exp_name)

    exp_main = build_experiment(
        cfg=cfg,
        init=init,
        ensemble_size=args.ensemble_size,
        sims_per_model=args.sims_per_model,
        max_steps=args.max_steps,
        sample_at_evaluate=args.sample_main_at_evaluate
    )

    exp_fast_by_pair = {}
    if args.use_fast_eval:
        print("-- preparing fast experiments")
        for ensemble_size_fast, sims_per_model_fast in fast_pairs:
            print(f"-- fast pair: ensemble_size_fast={ensemble_size_fast}, sims_per_model_fast={sims_per_model_fast}")
            exp_fast_by_pair[(ensemble_size_fast, sims_per_model_fast)] = build_experiment(
                cfg=cfg,
                init=init,
                ensemble_size=ensemble_size_fast,
                sims_per_model=sims_per_model_fast,
                max_steps=args.max_steps,
                sample_at_evaluate=True
            )

    output_nodes = list(cfg["target"].keys())
    all_moves = build_all_moves(cfg["ens_dir"], output_nodes)
    print(f"-- total moves available: {len(all_moves)}")

    tasks = []
    for algo in args.algos:
        for depth in args.depths:
            for timeout in args.timeouts:
                for trial in range(args.n_trials):
                    if algo == "BILNMCS":
                        for ensemble_size_fast, sims_per_model_fast in fast_pairs:
                            tasks.append((algo, depth, timeout, trial, ensemble_size_fast, sims_per_model_fast))
                    else:
                        tasks.append((algo, depth, timeout, trial, None, None))

    if not tasks:
        print("No tasks to run.")
        return

    local_tasks = [task for i, task in enumerate(tasks) if i % args.num_chunks == args.chunk_id]

    print(f"-- total tasks: {len(tasks)}")
    print(f"-- chunk {args.chunk_id}/{args.num_chunks}: {len(local_tasks)} tasks")

    results = []

    common_suffix = make_algo_suffixes(args)

    for algo, depth, timeout, trial, ensemble_size_fast, sims_per_model_fast in tqdm(
        local_tasks,
        desc=f"Chunk {args.chunk_id}",
        unit="run"
    ):
        try:
            seed = make_stable_seed(
                args.exp_name,
                algo,
                depth,
                timeout,
                trial,
                args.chunk_id,
                args.num_chunks,
                ensemble_size_fast,
                sims_per_model_fast,
                args.ensemble_size,
                args.sims_per_model,
                args.sample_main_at_evaluate,
                args.lnmcs_b if algo == "LNMCS" else None
            )
            random.seed(seed)
            np.random.seed(seed)

            trial_cache = {}
            trial_history = []
            t_trial_start = time.time()

            main_best_tracker = {"value": -float("inf"), "mutant_dict": None}
            fast_best_tracker = {"value": -float("inf"), "mutant_dict": None}

            eval_main_fn = make_tracked_eval_fn(
                exp=exp_main,
                all_moves=all_moves,
                cache=trial_cache,
                evaluator="main",
                trial_history=trial_history,
                t_start=t_trial_start,
                best_tracker=main_best_tracker,
                use_cache=not args.sample_main_at_evaluate,
                reject_duplicates=True
            )

            exp_fast = None
            eval_fast_fn = None
            actual_ensemble_size_fast = None

            if algo == "BILNMCS":
                exp_fast = exp_fast_by_pair[(ensemble_size_fast, sims_per_model_fast)]
                actual_ensemble_size_fast = exp_fast.actual_size

                eval_fast_fn = make_tracked_eval_fn(
                    exp=exp_fast,
                    all_moves=all_moves,
                    cache=trial_cache,
                    evaluator="fast",
                    trial_history=trial_history,
                    t_start=t_trial_start,
                    best_tracker=fast_best_tracker,
                    use_cache=False,
                    reject_duplicates=True
                )

            wall_clock_start = time.time()
            search_start = time.time()

            out = run_one_algo(
                algo=algo,
                depth=depth,
                timeout=timeout,
                all_moves=all_moves,
                eval_main_fn=eval_main_fn,
                eval_fast_fn=eval_fast_fn,
                nmcs_level=args.nmcs_level,
                lnmcs_level=args.lnmcs_level,
                nrpa_level=args.nrpa_level,
                gnrpa_level=args.gnrpa_level,
                gnrpa_N=args.gnrpa_N,
                gnrpa_tau=args.gnrpa_tau,
                bilnmcs_level=args.bilnmcs_level,
                bilnmcs_b=args.bilnmcs_b,
                bilnmcs_r=args.bilnmcs_r,
                bilnmcs_e=args.bilnmcs_e,
                lnmcs_b=args.lnmcs_b,
                lnmcs_r=args.lnmcs_r,
                lnmcs_e=args.lnmcs_e
            )

            search_elapsed = time.time() - search_start
            wall_clock_elapsed = time.time() - wall_clock_start

            mutant_dict = moves_to_mutant_dict(out["mutations"], all_moves)

            rechecked_main_score = exp_main.evaluate(mutant_dict)
            score_gap = abs(float(out["score"]) - float(rechecked_main_score))

            trial_history.append({
                "time_sec": search_elapsed,
                "score": rechecked_main_score,
                "evaluator": "main",
                "event_type": "final",
                "mutant_dict": str(mutant_dict)
            })

            if score_gap > 1e-9:
                print(
                    f"[WARN] final score mismatch | "
                    f"algo={algo} depth={depth} trial={trial} "
                    f"reported={out['score']} rechecked_main={rechecked_main_score}"
                )

            decoded_moves = [decode_single_move(m, all_moves) for m in out["mutations"]] if out["mutations"] else []

            results.append({
                "experiment": args.exp_name,
                "algorithm": algo,

                "ensemble_size": args.ensemble_size,
                "actual_ensemble_size": exp_main.actual_size,
                "sims_per_model": args.sims_per_model,
                "sample_main_at_evaluate": args.sample_main_at_evaluate,

                "ensemble_size_fast": ensemble_size_fast,
                "actual_ensemble_size_fast": actual_ensemble_size_fast,
                "sims_per_model_fast": sims_per_model_fast,

                "max_steps": args.max_steps,
                "depth": depth,
                "timeout": timeout,
                "trial": trial,
                "seed": seed,

                "score_reported_by_algo": out["score"],
                "score_rechecked_main": rechecked_main_score,
                "score_gap_abs": score_gap,

                "mutations": out["mutations"],
                "decoded_mutations": str(decoded_moves),
                "n_mutations_raw": len(out["mutations"]) if out["mutations"] is not None else 0,
                "n_mutations_effective": len(mutant_dict),
                "mutant_dict": mutant_dict,

                "wall_clock_elapsed_sec": wall_clock_elapsed,
                "search_elapsed_sec": search_elapsed,

                "eval_count": out["eval_count"],
                "playout_count": out["playout_count"],
                "main_eval_count": out["main_eval_count"],
                "fast_eval_count": out["fast_eval_count"],

                "algo_level": out["algo_level"],
                "algo_b": out["algo_b"],
                "algo_r": out["algo_r"],
                "algo_e": out["algo_e"],
                "algo_tau": out["algo_tau"],
                "algo_N": out["algo_N"],
            })

            if trial_history:
                hist_df = pd.DataFrame(trial_history)
                hist_df["experiment"] = args.exp_name
                hist_df["algorithm"] = algo

                hist_df["ensemble_size"] = args.ensemble_size
                hist_df["actual_ensemble_size"] = exp_main.actual_size
                hist_df["sims_per_model"] = args.sims_per_model
                hist_df["sample_main_at_evaluate"] = args.sample_main_at_evaluate

                hist_df["ensemble_size_fast"] = ensemble_size_fast
                hist_df["actual_ensemble_size_fast"] = actual_ensemble_size_fast
                hist_df["sims_per_model_fast"] = sims_per_model_fast

                hist_df["depth"] = depth
                hist_df["timeout"] = timeout
                hist_df["trial"] = trial
                hist_df["chunk_id"] = args.chunk_id
                hist_df["num_chunks"] = args.num_chunks
                hist_df["seed"] = seed

                hist_path = os.path.join(
                    args.outdir,
                    (
                        f"julia_algo_history_"
                        f"exp_{args.exp_name}_"
                        f"algo_{algo}_"
                        f"depth_{depth}_"
                        f"timeout_{timeout}_"
                        f"{common_suffix}_"
                        f"fastens_{ensemble_size_fast}_"
                        f"fastsims_{sims_per_model_fast}_"
                        f"trial_{trial}_"
                        f"chunk_{args.chunk_id}_of_{args.num_chunks}.csv"
                    )
                )
                hist_df.to_csv(hist_path, index=False)

        except Exception as e:
            print(f"[ERROR] Task {(algo, depth, timeout, trial, ensemble_size_fast, sims_per_model_fast)}: {e}")

    if not results:
        print("No results were produced.")
        return

    df = pd.DataFrame(results)

    algos_flag = "-".join(sorted(set(args.algos)))
    timeouts_flag = "-".join(map(str, sorted(set(args.timeouts))))

    fast_pairs_flag = "none"
    if args.use_fast_eval:
        fast_pairs_flag = "__".join(
            [f"ens{ens}_sim{sims}" for ens, sims in fast_pairs]
        )

    out_all = os.path.join(
        args.outdir,
        f"julia_algo_results_"
        f"exp_{args.exp_name}_"
        f"algos_{algos_flag}_"
        f"{common_suffix}_"
        f"timeouts_{timeouts_flag}_"
        f"fastpairs_{fast_pairs_flag}_"
        f"chunk_{args.chunk_id}_of_{args.num_chunks}.csv"
    )
    df.to_csv(out_all, index=False)
    print(f"-- saved all trials: {out_all}")

    summary = df.groupby(
        [
            "experiment",
            "algorithm",
            "depth",
            "timeout",

            "ensemble_size",
            "actual_ensemble_size",
            "sims_per_model",
            "sample_main_at_evaluate",

            "ensemble_size_fast",
            "actual_ensemble_size_fast",
            "sims_per_model_fast",

            "algo_level",
            "algo_b",
            "algo_r",
            "algo_e",
            "algo_tau",
            "algo_N",
        ],
        as_index=False
    ).agg(
        n_trials=("trial", "count"),
        mean_score_reported_by_algo=("score_reported_by_algo", "mean"),
        mean_score_rechecked_main=("score_rechecked_main", "mean"),
        best_score_rechecked_main=("score_rechecked_main", "max"),
        mean_score_gap_abs=("score_gap_abs", "mean"),
        mean_search_elapsed_sec=("search_elapsed_sec", "mean"),
        mean_wall_clock_elapsed_sec=("wall_clock_elapsed_sec", "mean"),
        mean_eval_count=("eval_count", "mean"),
        mean_playout_count=("playout_count", "mean"),
        mean_main_eval_count=("main_eval_count", "mean"),
        mean_fast_eval_count=("fast_eval_count", "mean"),
    )

    out_summary = os.path.join(
        args.outdir,
        f"julia_algo_summary_"
        f"exp_{args.exp_name}_"
        f"algos_{algos_flag}_"
        f"{common_suffix}_"
        f"timeouts_{timeouts_flag}_"
        f"fastpairs_{fast_pairs_flag}_"
        f"chunk_{args.chunk_id}_of_{args.num_chunks}.csv"
    )
    summary.to_csv(out_summary, index=False)
    print(f"-- saved summary: {out_summary}")


if __name__ == "__main__":
    main()