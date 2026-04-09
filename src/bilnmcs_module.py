#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time


# ------------------ Helper Functions ------------------ #

def normalize_sorted_list(state_list):
    """Return a NEW sorted list of (gene, bool) pairs."""
    return sorted(state_list, key=lambda x: (x[0], x[1]))


def normalize_key(state_list):
    """Hashable, order-independent cache key."""
    return tuple(normalize_sorted_list(state_list))


def is_terminal(state_set, depth):
    """Check if the mutation set has reached the desired size."""
    return len(state_set) >= depth


def legal_moves_fn(state_set, all_moves):
    """Return list of mutations not yet applied. Preserves order of all_moves."""
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]


def random_playout(state_set, all_moves, depth, evaluate_fn):
    """Complete the mutation set randomly up to `depth` and score it."""
    remaining = legal_moves_fn(state_set, all_moves)
    k = depth - len(state_set)
    k = max(0, min(k, len(remaining)))
    tail = random.sample(remaining, k=k)
    full_set = normalize_sorted_list(list(state_set) + tail)
    score = evaluate_fn(full_set)
    return score, full_set


# ------------------ Bi-LNMCS ------------------ #

def bilnmcs(state, level, depth, all_moves, evaluate_main_fn, evaluate_fast_fn, *,
            b=2, r=0.5, e=None, timeout_sec=None):
    """
    Bi-Lazy NMCS with:
    - main evaluator for real scoring
    - fast evaluator for cheap filtering

    Retour
    ------
    best_score, best_state, main_eval_count, fast_eval_count
    """
    deadline = time.time() + timeout_sec if timeout_sec else None

    max_depth = depth
    tr = [{'mean': 0.0, 'count': 0} for _ in range(max_depth + 1)]
    trmax = [float('-inf') for _ in range(max_depth + 1)]

    c_main = [dict() for _ in range(level + 1)]

    best = {'score': -float('inf'), 'state': []}
    main_eval_counter = [0]
    fast_eval_counter = [0]

    def counted_main_fn(state):
        main_eval_counter[0] += 1
        return evaluate_main_fn(state)

    def counted_fast_fn(state):
        fast_eval_counter[0] += 1
        return evaluate_fast_fn(state)

    score, state_out = _bilnmcs(
        state=state,
        level=level,
        depth=depth,
        all_moves=all_moves,
        evaluate_main_fn=counted_main_fn,
        evaluate_fast_fn=counted_fast_fn,
        c_main=c_main,
        deadline=deadline,
        best=best,
        tr=tr,
        trmax=trmax,
        b=b,
        r=r,
        e=e
    )

    return score, normalize_sorted_list(state_out), main_eval_counter[0], fast_eval_counter[0]


def _bilnmcs(state, level, depth, all_moves, evaluate_main_fn, evaluate_fast_fn, *,
             c_main, deadline, best, tr, trmax, b, r, e):
    if deadline and time.time() > deadline:
        return best['score'], best['state']

    state = normalize_sorted_list(list(state))
    key = normalize_key(state)

    if key in c_main[level]:
        return c_main[level][key]

    if level == 0 or is_terminal(state, depth):
        if key in c_main[0]:
            score, state_out = c_main[0][key]
        else:
            score, state_out = random_playout(state, all_moves, depth, evaluate_main_fn)
            c_main[0][key] = (score, state_out)

        if score > best['score']:
            best.update(score=score, state=state_out)

        return score, state_out

    best_score_level = -float('inf')
    best_set_level = []
    state_cur = list(state)

    while not is_terminal(state_cur, depth):
        if deadline and time.time() > deadline:
            return best['score'], best['state']

        moves = legal_moves_fn(state_cur, all_moves)
        if not moves:
            break

        if e is not None and len(moves) > e:
            moves = random.sample(moves, e)

        d = min(len(state_cur), depth)
        candidates = []

        for move in moves:
            if deadline and time.time() > deadline:
                return best['score'], best['state']

            child_state = normalize_sorted_list(state_cur + [move])

            total = 0.0
            n_playouts = max(1, b)

            # Pas de cache fast ici :
            # chaque playout doit pouvoir rééchantillonner les BN via evaluate_fast_fn
            for _ in range(n_playouts):
                score, _ = random_playout(child_state, all_moves, depth, evaluate_fast_fn)
                total += score

            mean_eval = total / n_playouts
            candidates.append((mean_eval, move))

            acc = tr[d]
            acc['mean'] = (acc['mean'] * acc['count'] + mean_eval) / (acc['count'] + 1)
            acc['count'] += 1

            if mean_eval > trmax[d]:
                trmax[d] = mean_eval

        mu_d = tr[d]['mean']
        best_d = trmax[d]
        theta = mu_d + r * (best_d - mu_d)

        local_best_score = -float('inf')
        local_best_set = []

        for mean_eval, move in candidates:
            if deadline and time.time() > deadline:
                return best['score'], best['state']

            child_state = normalize_sorted_list(state_cur + [move])
            child_key = normalize_key(child_state)
            next_level = 0 if mean_eval < theta else (level - 1)

            if next_level == 0:
                if child_key in c_main[0]:
                    score, state_out = c_main[0][child_key]
                else:
                    score, state_out = random_playout(child_state, all_moves, depth, evaluate_main_fn)
                    c_main[0][child_key] = (score, state_out)
            else:
                if child_key in c_main[level - 1]:
                    score, state_out = c_main[level - 1][child_key]
                else:
                    score, state_out = _bilnmcs(
                        child_state,
                        level - 1,
                        depth,
                        all_moves,
                        evaluate_main_fn,
                        evaluate_fast_fn,
                        c_main=c_main,
                        deadline=deadline,
                        best=best,
                        tr=tr,
                        trmax=trmax,
                        b=b,
                        r=r,
                        e=e
                    )
                    c_main[level - 1][child_key] = (score, state_out)

            if score > local_best_score:
                local_best_score = score
                local_best_set = state_out

            if score > best_score_level:
                best_score_level = score
                best_set_level = state_out

            # best global mis à jour seulement avec les scores main
            if score > best['score']:
                best.update(score=score, state=state_out)

        next_elem = next((x for x in local_best_set if x not in state_cur), None)
        if next_elem is None:
            break

        state_cur = normalize_sorted_list(state_cur + [next_elem])

    if best_set_level:
        return best_score_level, best_set_level

    return best['score'], best['state']