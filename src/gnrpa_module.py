#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import time


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def legal_moves_fn(state_set, all_moves):
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]


def code_fn(state, move):
    return hash((tuple(sorted(state)), move))


def softmax_probs(logits, tau):
    expv = [math.exp(l / tau) for l in logits]
    Z = sum(expv)
    return [v / Z for v in expv], Z


# ----------------------------------------------------------------------------
# Level-0 playout with trace recording
# ----------------------------------------------------------------------------

def gnrpa_playout_and_trace(policy, bias, tau, depth, evaluate_fn, all_moves):
    state = []
    sequence = []

    code_matrix = []
    index_list = []
    O_list = []
    Z_list = []

    while len(state) < depth:
        applied_genes = {gene for gene, _ in state}
        moves = [m for m in all_moves if m[0] not in applied_genes]
        if not moves:
            break

        codes = [code_fn(state, m) for m in moves]
        ws = [policy.get(c, 0.0) for c in codes]
        bs = [bias.get(c, 0.0) for c in codes]

        o = [math.exp((w + b) / tau) for w, b in zip(ws, bs)]
        Z = sum(o)
        if Z == 0.0:
            break

        chosen_j = random.choices(range(len(o)), weights=o)[0]

        code_matrix.append(codes)
        index_list.append(chosen_j)
        O_list.append(o)
        Z_list.append(Z)

        chosen_move = moves[chosen_j]
        state.append(chosen_move)
        sequence.append(chosen_move)

    sequence = sorted(sequence)
    score = evaluate_fn(sequence)
    return score, sequence, code_matrix, index_list, O_list, Z_list


def gnrpa_adapt_inplace(policy, code_matrix, index_list, O_list, Z_list, tau):
    factor = tau
    for i, codes_i in enumerate(code_matrix):
        best_j = index_list[i]
        oi = O_list[i]
        zi = Z_list[i]

        for j, c in enumerate(codes_i):
            policy.setdefault(c, 0.0)
            pij = oi[j] / zi
            delta = pij - (1.0 if j == best_j else 0.0)
            policy[c] -= factor * delta

    return policy


# ----------------------------------------------------------------------------
# GNRPA
# ----------------------------------------------------------------------------

def gnrpa(level, policy, bias, tau, depth, evaluate_fn, all_moves, N=100, timeout_sec=None):
    """
    GNRPA avec fonction d'évaluation directe.

    Retour
    ------
    best_score, best_seq, code_matrix, index_list, O_list, Z_list, eval_count
    """
    deadline = time.time() + timeout_sec if timeout_sec else None
    eval_counter = [0]

    def counted_evaluate_fn(state):
        eval_counter[0] += 1
        return evaluate_fn(state)

    score, seq, cm, il, ol, zl = _gnrpa(
        level=level,
        policy=policy,
        bias=bias,
        tau=tau,
        depth=depth,
        evaluate_fn=counted_evaluate_fn,
        all_moves=all_moves,
        N=N,
        deadline=deadline
    )

    return score, seq, cm, il, ol, zl, eval_counter[0]


def _gnrpa(level, policy, bias, tau, depth, evaluate_fn, all_moves, N, deadline):
    if deadline and time.time() > deadline:
        return float("-inf"), [], [], [], [], []

    if level == 0:
        return gnrpa_playout_and_trace(policy, bias, tau, depth, evaluate_fn, all_moves)

    best_score = float("-inf")
    best_seq = []
    best_traces = None

    for _ in range(N):
        if deadline and time.time() > deadline:
            break

        child_policy = policy.copy()
        sc, seq, cm, il, ol, zl = _gnrpa(
            level=level - 1,
            policy=child_policy,
            bias=bias,
            tau=tau,
            depth=depth,
            evaluate_fn=evaluate_fn,
            all_moves=all_moves,
            N=N,
            deadline=deadline
        )

        if sc > best_score:
            best_score = sc
            best_seq = seq
            best_traces = (cm, il, ol, zl)

        if best_traces is not None:
            gnrpa_adapt_inplace(policy, *best_traces, tau)

    if best_traces is not None:
        cm, il, ol, zl = best_traces
    else:
        cm, il, ol, zl = [], [], [], []

    return best_score, best_seq, cm, il, ol, zl