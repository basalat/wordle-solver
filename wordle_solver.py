"""Validate completed engine candidates before expensive campaign evaluation.

Uses final_beams only. There is deliberately no fallback to best_state:
that value may be an incomplete parent returned after unsuccessful branching.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from core.topological_state import StateVerdict, TopologicalState

from .fly_pocket_token_geometry import (
    STANDARD_AA,
    reference_sequence as validate_reference_sequence,
    residue_indices,
)


@dataclass(frozen=True)
class FlyPocketCandidateBatch:
    states: tuple[TopologicalState, ...]
    rejection_counts: dict[str, int]
    failure_reason: str | None
    collapse_reason: str | None

    @property
    def sequences(self) -> tuple[str, ...]:
        return tuple(state.substrate for state in self.states)

    @property
    def successful(self) -> bool:
        return bool(self.states)


def extract_fly_pocket_candidates(
    search_result: Mapping,
    *,
    reference_sequence: str,
    mutation_targets: Iterable[int],
    seen_sequences: Iterable[str] = (),
) -> FlyPocketCandidateBatch:
    """Return completed, retained, unique candidates.

    Policy:
    - A reported collapse makes the attempt unsuccessful.
    - Missing/malformed result structure raises a contract error.
    - Candidate-level validation failures are counted.
    - PENDING is the retained verdict in the supplied ManifoldEngine.
    - Deduplication keeps the highest-ranked valid occurrence.
    - seen_sequences is read but never modified.
    """
    reference = validate_reference_sequence(reference_sequence)
    targets = residue_indices(mutation_targets)

    if not targets or max(targets) >= len(reference):
        raise ValueError(
            "mutation_targets must be nonempty and inside the reference"
        )

    if not isinstance(search_result, Mapping):
        raise TypeError("search_result must be a mapping")
    if "final_beams" not in search_result:
        raise ValueError("search_result is missing final_beams")

    beams = search_result["final_beams"]
    if not isinstance(beams, (list, tuple)):
        raise TypeError("final_beams must be a list or tuple")
    if any(not isinstance(state, TopologicalState) for state in beams):
        raise TypeError("final_beams must contain TopologicalState objects")

    if isinstance(seen_sequences, str):
        raise TypeError("seen_sequences must be an iterable of sequences")
    seen = set(seen_sequences)
    if any(not isinstance(sequence, str) for sequence in seen):
        raise TypeError("seen_sequences must contain strings")

    collapse = search_result.get("collapse_reason")
    if collapse is not None:
        if not isinstance(collapse, str):
            raise TypeError("collapse_reason must be a string or None")
        return FlyPocketCandidateBatch(
            states=(),
            rejection_counts={},
            failure_reason="search_collapsed",
            collapse_reason=collapse,
        )

    rejected = Counter()
    valid = []

    for state in beams:
        sequence = state.substrate

        if state.verdict != StateVerdict.PENDING:
            rejected["unretained_verdict"] += 1
            continue

        if not isinstance(sequence, str):
            rejected["invalid_substrate"] += 1
            continue

        if len(sequence) != len(reference):
            rejected["wrong_length"] += 1
            continue

        if "_" in sequence:
            rejected["incomplete_sequence"] += 1
            continue

        if set(sequence) - set(STANDARD_AA):
            rejected["noncanonical_sequence"] += 1
            continue

        if any(
            sequence[position] != original
            for position, original in enumerate(reference)
            if position not in targets
        ):
            rejected["protected_residue_changed"] += 1
            continue

        try:
            rank = float(state.sort_metric)
        except (TypeError, ValueError, OverflowError):
            rejected["invalid_rank"] += 1
            continue

        if not math.isfinite(rank):
            rejected["invalid_rank"] += 1
            continue

        valid.append((rank, state))

    valid.sort(key=lambda item: item[0], reverse=True)

    accepted = []
    for _, state in valid:
        sequence = state.substrate

        if sequence in seen:
            rejected["duplicate_sequence"] += 1
            continue

        seen.add(sequence)
        accepted.append(state)

    if accepted:
        failure = None
    elif not beams:
        failure = "no_retained_states"
    elif len(rejected) == 1:
        failure = next(iter(rejected))
    else:
        failure = "no_valid_completed_candidates"

    return FlyPocketCandidateBatch(
        states=tuple(accepted),
        rejection_counts=dict(rejected),
        failure_reason=failure,
        collapse_reason=None,
    )
