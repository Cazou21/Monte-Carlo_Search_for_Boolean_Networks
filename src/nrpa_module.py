#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import copy
import math
import random


def is_terminal(state_set, depth):
    return len(state_set) >= depth


def legal_moves_fn(state_set, all_moves):
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]


def code(state, move):
    return hash((tuple(sorted(state)), move))


def random_move(state_set, all_moves, policy):
    moves = legal_moves_fn(state_set, all_moves)
    if not moves:
        raise RuntimeError("No legal moves available during playout.")

    weights = [math.exp(policy.get(code(state_set, m), 0.0)) for m in moves]
    total = sum(weights)

    stop = random.random() * total
    cumulative = 0.0
    for move, weight in zip(moves, weights):
        cumulative += weight
        if cumulative >= stop:
            return move

    return moves[-1]


def play(state, move):
    return state + [move]


def playout(state, policy, depth, evaluate_fn, all_moves, playout_counter=None):
    state = list(state)

    while not is_terminal(state, depth):
        move = random_move(state, all_moves, policy)
        state = play(state, move)

    state = sorted(state)

    if playout_counter is not None:
        playout_counter[0] += 1

    return evaluate_fn(state), state


def adapt(policy, state_set, all_moves):
    new_policy = copy.deepcopy(policy)
    state = []

    for move in state_set:
        moves = legal_moves_fn(state, all_moves)
        Z = sum(math.exp(policy.get(code(state, m), 0.0)) for m in moves)

        for m in moves:
            c = code(state, m)
            prob = math.exp(policy.get(c, 0.0)) / Z
            new_policy[c] = new_policy.get(c, 0.0) - prob

        c_best = code(state, move)
        new_policy[c_best] = new_policy.get(c_best, 0.0) + 1.0
        state = play(state, move)

    return new_policy


def nrpa(level, policy, depth, evaluate_fn, all_moves, timeout_sec=None):
    """
    NRPA avec fonction d'évaluation directe.

    Paramètres
    ----------
    level : int
        Niveau de récursion NRPA.
    policy : dict
        Politique initiale.
    depth : int
        Nombre de mutations à construire.
    evaluate_fn : callable
        Fonction qui prend un état (liste de moves) et retourne un score.
    all_moves : list
        Liste de tous les moves possibles.
    timeout_sec : int | float | None
        Temps limite en secondes.

    Retour
    ------
    best_score, best_state, playout_count, eval_count
    """
    deadline = time.time() + timeout_sec if timeout_sec else None
    best_global = {"score": -float("inf"), "state": []}
    playout_counter = [0]
    eval_counter = [0]

    def counted_evaluate_fn(state):
        eval_counter[0] += 1
        return evaluate_fn(state)

    _nrpa(
        level=level,
        policy=policy,
        depth=depth,
        evaluate_fn=counted_evaluate_fn,
        all_moves=all_moves,
        deadline=deadline,
        best_global=best_global,
        playout_counter=playout_counter,
    )

    return best_global["score"], best_global["state"], playout_counter[0], eval_counter[0]


def _nrpa(level, policy, depth, evaluate_fn, all_moves, deadline, best_global, playout_counter):
    if deadline and time.time() > deadline:
        return best_global["score"], best_global["state"]

    if level == 0:
        score, state = playout(
            state=[],
            policy=policy,
            depth=depth,
            evaluate_fn=evaluate_fn,
            all_moves=all_moves,
            playout_counter=playout_counter,
        )

        if score > best_global["score"]:
            best_global["score"] = score
            best_global["state"] = state

        return score, state

    best = -float("inf")
    best_state = []
    seen_states = set()

    while True:
        if deadline and time.time() > deadline:
            break

        pol = policy.copy()
        score, state = _nrpa(
            level=level - 1,
            policy=pol,
            depth=depth,
            evaluate_fn=evaluate_fn,
            all_moves=all_moves,
            deadline=deadline,
            best_global=best_global,
            playout_counter=playout_counter,
        )

        state_tuple = tuple(state)

        if state_tuple in seen_states:
            break
        seen_states.add(state_tuple)

        if score > best:
            best = score
            best_state = state

        policy = adapt(policy, best_state, all_moves)

    return best_global["score"], best_global["state"]