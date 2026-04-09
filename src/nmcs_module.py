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


# ------------------ NMCS Core ------------------ #

def nmcs(state, level, depth, best_moves, evaluate_fn, timeout_sec=None):
    """
    NMCS avec fonction d'évaluation directe.

    Paramètres
    ----------
    state : list
        État initial (liste de moves).
    level : int
        Niveau de récursion NMCS.
    depth : int
        Taille cible de la séquence de mutations.
    best_moves : list
        Liste des moves possibles.
    evaluate_fn : callable
        Fonction qui prend un état (liste de moves) et retourne un score.
    timeout_sec : int | float | None
        Temps limite en secondes.

    Retour
    ------
    best_score, best_state, eval_count
    """
    deadline = time.time() + timeout_sec if timeout_sec else None
    best = {"score": -float("inf"), "state": []}
    caches = [dict() for _ in range(level + 1)]
    eval_counter = [0]

    def counted_evaluate_fn(state):
        eval_counter[0] += 1
        return evaluate_fn(state)

    score, state_out = _nmcs(
        state=state,
        level=level,
        depth=depth,
        best_moves=best_moves,
        evaluate_fn=counted_evaluate_fn,
        caches=caches,
        deadline=deadline,
        best=best
    )

    return score, normalize_sorted_list(state_out), eval_counter[0]


def _nmcs(state, level, depth, best_moves, evaluate_fn, caches, deadline, best):
    if deadline and time.time() > deadline:
        return best["score"], best["state"]

    key = normalize_key(state)

    if key in caches[level]:
        return caches[level][key]

    if level == 0 or is_terminal(state, depth):
        if key in caches[0]:
            score, state_out = caches[0][key]
        else:
            score, state_out = random_playout(state, best_moves, depth, evaluate_fn)
            caches[0][key] = (score, state_out)

        if score > best["score"]:
            best["score"] = score
            best["state"] = state_out

        return score, state_out

    state_cur = normalize_sorted_list(list(state))
    best_score_here = -float("inf")
    best_set_here = []

    while not is_terminal(state_cur, depth):
        if deadline and time.time() > deadline:
            return best["score"], best["state"]

        moves = legal_moves_fn(state_cur, best_moves)
        local_best_score = -float("inf")
        local_best_set = []

        for move in moves:
            if deadline and time.time() > deadline:
                return best["score"], best["state"]

            child_state = normalize_sorted_list(state_cur + [move])
            child_key = normalize_key(child_state)

            if child_key in caches[level - 1]:
                score, state_out = caches[level - 1][child_key]
            else:
                score, state_out = _nmcs(
                    state=child_state,
                    level=level - 1,
                    depth=depth,
                    best_moves=best_moves,
                    evaluate_fn=evaluate_fn,
                    caches=caches,
                    deadline=deadline,
                    best=best
                )
                caches[level - 1][child_key] = (score, state_out)

            if score > local_best_score:
                local_best_score = score
                local_best_set = state_out

            if score > best["score"]:
                best["score"] = score
                best["state"] = state_out

        if local_best_score > best_score_here:
            best_score_here = local_best_score
            best_set_here = local_best_set

        next_elem = None
        for x in best_set_here:
            if x not in state_cur:
                next_elem = x
                break

        if next_elem is None:
            break

        state_cur = normalize_sorted_list(state_cur + [next_elem])

    return best_score_here, state_cur