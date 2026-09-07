"""Prevalence-aware binary metrics and deterministic threshold selection."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .contracts import THRESHOLDS, validate_probability


@dataclass(frozen=True, slots=True)
class Metrics:
    threshold: float
    prevalence: float
    positives: int
    negatives: int
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def binary_metrics(
    truth: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    prevalence: float,
) -> Metrics:
    if len(truth) != len(scores) or not truth:
        raise ValueError("truth and scores must have the same non-zero length")
    cutoff = validate_probability(threshold, name="threshold")
    prevalence = float(prevalence)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between 0 and 1")

    tp = fp = tn = fn = 0
    for label, raw_score in zip(truth, scores, strict=True):
        if label not in (0, 1):
            raise ValueError("truth values must be 0 or 1")
        predicted = validate_probability(raw_score) > cutoff
        if label == 1 and predicted:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    return _metrics_from_counts(
        threshold=cutoff,
        prevalence=prevalence,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def _metrics_from_counts(
    *,
    threshold: float,
    prevalence: float,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Metrics:
    positives = tp + fn
    negatives = fp + tn
    if positives == 0 or negatives == 0:
        raise ValueError("metrics require at least one positive and one negative")

    tpr = tp / positives
    fpr = fp / negatives
    expected_tp = prevalence * tpr
    expected_fp = (1.0 - prevalence) * fpr
    expected_fn = prevalence * (1.0 - tpr)
    precision_denominator = expected_tp + expected_fp
    precision = expected_tp / precision_denominator if precision_denominator else 0.0
    recall = tpr
    f1_denominator = 2.0 * expected_tp + expected_fp + expected_fn
    f1 = 2.0 * expected_tp / f1_denominator if f1_denominator else 0.0
    return Metrics(
        threshold=threshold,
        prevalence=prevalence,
        positives=positives,
        negatives=negatives,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def threshold_curve(
    truth: Sequence[int],
    scores: Sequence[float],
    *,
    prevalence: float,
    thresholds: Iterable[float] = THRESHOLDS,
) -> list[Metrics]:
    if len(truth) != len(scores) or not truth:
        raise ValueError("truth and scores must have the same non-zero length")
    prevalence = float(prevalence)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between 0 and 1")
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for label, raw_score in zip(truth, scores, strict=True):
        if label not in (0, 1):
            raise ValueError("truth values must be 0 or 1")
        score = validate_probability(raw_score)
        (positive_scores if label == 1 else negative_scores).append(score)
    if not positive_scores or not negative_scores:
        raise ValueError("metrics require at least one positive and one negative")
    positive_scores.sort()
    negative_scores.sort()
    cutoffs = [validate_probability(value, name="threshold") for value in thresholds]
    if not cutoffs:
        raise ValueError("thresholds is empty")
    positives = len(positive_scores)
    negatives = len(negative_scores)
    curve = []
    for cutoff in cutoffs:
        # Strict p_euk > threshold means scores equal to the cutoff remain
        # negative, hence bisect_right rather than bisect_left.
        tp = positives - bisect_right(positive_scores, cutoff)
        fp = negatives - bisect_right(negative_scores, cutoff)
        curve.append(
            _metrics_from_counts(
                threshold=cutoff,
                prevalence=prevalence,
                tp=tp,
                fp=fp,
                tn=negatives - fp,
                fn=positives - tp,
            )
        )
    return curve


def select_best_threshold(curve: Sequence[Metrics]) -> Metrics:
    """Maximize F1, then precision, then minimize FPR, then raise threshold."""

    if not curve:
        raise ValueError("threshold curve is empty")
    return max(curve, key=_threshold_selection_key)


def _threshold_selection_key(row: Metrics) -> tuple[float, float, float, float]:
    return (row.f1, row.precision, -row.fpr, row.threshold)


def select_best_threshold_exact(
    truth: Sequence[int],
    scores: Sequence[float],
    *,
    prevalence: float,
) -> Metrics:
    """Find the exact best threshold over the complete closed interval [0, 1].

    With the locked strict rule ``score > threshold``, predictions can change
    only when the threshold crosses an observed score.  Evaluating 0, 1, and
    every distinct observed score is therefore exact; no arbitrary grid or
    monotonic score transform is needed.  The scan stores only the current and
    best count tables rather than a potentially very large threshold curve.
    """

    if len(truth) != len(scores) or not truth:
        raise ValueError("truth and scores must have the same non-zero length")
    prevalence = float(prevalence)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between 0 and 1")

    pairs: list[tuple[float, int]] = []
    positives = negatives = 0
    tp = fp = 0
    for label, raw_score in zip(truth, scores, strict=True):
        if label not in (0, 1):
            raise ValueError("truth values must be 0 or 1")
        score = validate_probability(raw_score)
        pairs.append((score, label))
        if label == 1:
            positives += 1
            tp += score > 0.0
        else:
            negatives += 1
            fp += score > 0.0
    if positives == 0 or negatives == 0:
        raise ValueError("metrics require at least one positive and one negative")

    pairs.sort(key=lambda item: item[0])
    best = _metrics_from_counts(
        threshold=0.0,
        prevalence=prevalence,
        tp=tp,
        fp=fp,
        tn=negatives - fp,
        fn=positives - tp,
    )

    index = 0
    while index < len(pairs):
        cutoff = pairs[index][0]
        group_positive = group_negative = 0
        while index < len(pairs) and pairs[index][0] == cutoff:
            if pairs[index][1] == 1:
                group_positive += 1
            else:
                group_negative += 1
            index += 1
        if cutoff == 0.0:
            # Zero-valued scores were already excluded from the threshold=0
            # initialization above.
            continue
        tp -= group_positive
        fp -= group_negative
        candidate = _metrics_from_counts(
            threshold=cutoff,
            prevalence=prevalence,
            tp=tp,
            fp=fp,
            tn=negatives - fp,
            fn=positives - tp,
        )
        if _threshold_selection_key(candidate) > _threshold_selection_key(best):
            best = candidate

    # If no observed score equals 1, threshold=max(score) and threshold=1 have
    # identical predictions.  The documented tie-break prefers the higher
    # threshold, so explicitly consider the right endpoint.
    endpoint = _metrics_from_counts(
        threshold=1.0,
        prevalence=prevalence,
        tp=0,
        fp=0,
        tn=negatives,
        fn=positives,
    )
    if _threshold_selection_key(endpoint) > _threshold_selection_key(best):
        best = endpoint
    return best
