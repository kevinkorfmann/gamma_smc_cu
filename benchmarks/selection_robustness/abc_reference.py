"""Transparent rejection-ABC reference and held-out model-choice diagnostics.

This is statistical infrastructure, not a validated regional analysis. Within
each model, bank rows must be independent draws from the declared parameter
prior. Adaptive proposals require importance weights and are NOT supported.
Returned support conditions on the supplied summaries and a finite tolerance;
it is not a full-data posterior or a goodness-of-fit probability.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import digest, write_json


def model_weights(labels, models, prior):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Require a nonempty one-dimensional model label array")
    if set(labels.tolist()) != set(models):
        raise ValueError("Every declared model must occur; unknown models are invalid")
    weights = np.zeros(len(labels))
    for model, probability in zip(models, prior):
        mask = labels == model
        weights[mask] = probability / mask.sum()
    return weights


def validate_groups(train_ids, heldout_ids):
    """Groups identify shared ancestry/replicates, not pairs or method outputs."""
    a, b = np.asarray(train_ids), np.asarray(heldout_ids)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0:
        raise ValueError("Require nonempty group ID vectors")
    if len(set(a.tolist())) != len(a) or len(set(b.tolist())) != len(b):
        raise ValueError("Aggregate linked observations to one row per simulation group")
    if set(a.tolist()) & set(b.tolist()):
        raise ValueError("Training and held-out simulations share group IDs")


class RejectionABC:
    def __init__(self, summaries, labels, model_prior, shrinkage=0.1):
        self.models = list(model_prior)
        self.prior = np.array(list(model_prior.values()), dtype=float)
        if (len(self.models) < 2 or not np.all(np.isfinite(self.prior))
                or np.any(self.prior <= 0) or not np.isclose(self.prior.sum(), 1)):
            raise ValueError("Require at least two models and positive probabilities summing to one")
        if not 0 < shrinkage <= 1:
            raise ValueError("Covariance shrinkage must be in (0,1]")
        self.labels = np.asarray(labels)
        self.X = np.asarray(summaries, dtype=float)
        if (self.X.ndim != 2 or self.X.shape[1] == 0 or len(self.X) < 3
                or len(self.X) != len(self.labels) or not np.all(np.isfinite(self.X))):
            raise ValueError("Require finite training summaries with one row per simulation")
        self.weights = model_weights(self.labels, self.models, self.prior)
        # Fit scaling/covariance using only the training prior predictive bank.
        self.center = self.weights @ self.X
        centered = self.X - self.center
        scale = np.sqrt(self.weights @ (centered ** 2))
        self.scale = np.where(scale > 0, scale, 1.0)
        Z = centered / self.scale
        covariance = Z.T @ (self.weights[:, None] * Z)
        covariance = (1 - shrinkage) * covariance + shrinkage * np.eye(Z.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        self.Z = Z @ self.whitener

    def infer(self, summary, fraction=0.02):
        summary = np.asarray(summary, dtype=float)
        if summary.shape != (self.X.shape[1],) or not np.all(np.isfinite(summary)):
            raise ValueError("Observed summaries must be finite and match the training schema")
        if not 0 < fraction <= 1:
            raise ValueError("Acceptance fraction must be in (0,1]")
        target = ((summary - self.center) / self.scale) @ self.whitener
        distances = np.linalg.norm(self.Z - target, axis=1)
        order = np.argsort(distances, kind="stable")
        cumulative = np.cumsum(self.weights[order])
        cutoff_index = min(np.searchsorted(cumulative, fraction), len(order) - 1)
        cutoff = distances[order[cutoff_index]]
        # Accept all ties. Breaking ties by row order can favor one model.
        indices = np.flatnonzero(distances <= cutoff)
        weight = self.weights[indices].copy()
        accepted_mass = float(weight.sum())
        weight /= accepted_mass
        support = np.array([weight[self.labels[indices] == m].sum() for m in self.models])
        return dict(model_support=support, indices=indices, weights=weight,
                    distances=distances[indices], epsilon=float(cutoff),
                    nearest_distance=float(distances.min()),
                    accepted_prior_mass=accepted_mass,
                    effective_sample_size=float(1 / np.sum(weight ** 2)))


def heldout_diagnostics(abc, summaries, labels, fraction=0.02):
    X, labels = np.asarray(summaries, dtype=float), np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("Invalid held-out shapes")
    weights = model_weights(labels, abc.models, abc.prior)
    results = [abc.infer(x, fraction) for x in X]
    probabilities = np.array([r["model_support"] for r in results])
    predicted = np.argmax(probabilities, axis=1)
    truth = np.array([abc.models.index(m) for m in labels])
    confusion = np.zeros((len(abc.models), len(abc.models)))
    for j in range(len(abc.models)):
        confusion[j] = np.bincount(predicted[truth == j], minlength=len(abc.models)) / np.sum(truth == j)
    onehot = np.eye(len(abc.models))[truth]
    reliability = []
    # One-vs-rest reliability, not just accuracy of the winning model.
    for j, model in enumerate(abc.models):
        bins = np.minimum((probabilities[:, j] * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if np.any(mask):
                w = weights[mask] / weights[mask].sum()
                reliability.append(dict(model=model, bin=b, count=int(mask.sum()),
                                        mean_support=float(w @ probabilities[mask, j]),
                                        observed_fraction=float(w @ onehot[mask, j])))
    return dict(models=abc.models, confusion_rows_true_columns_predicted=confusion.tolist(),
                prior_weighted_accuracy=float(weights @ (predicted == truth)),
                multiclass_brier=float(weights @ np.sum((probabilities - onehot)**2, axis=1)),
                reliability=reliability,
                nearest_distance_quantiles=dict(zip(["0.5", "0.95", "0.99"],
                    weighted_quantile(np.array([r["nearest_distance"] for r in results]), weights,
                                      [.5, .95, .99]).tolist())),
                probabilities=probabilities, nearest_distances=np.array([r["nearest_distance"] for r in results]))


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    cumulative /= cumulative[-1]
    return np.asarray(values)[order][np.minimum(np.searchsorted(cumulative, probabilities), len(order)-1)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--model-prior", type=Path, required=True, help='JSON object, e.g. {"neutral":0.5,"bgs":0.5}')
    p.add_argument("--observed", type=Path, help="NPZ summaries vector and summary_names; optional")
    p.add_argument("--fraction", type=float, default=.02)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    train, heldout = [np.load(p, allow_pickle=False) for p in [args.train, args.heldout]]
    for bank in [train, heldout]:
        if len(bank["group_id"]) != len(bank["summaries"]):
            raise ValueError("Each row requires a group ID")
        names = bank["summary_names"]
        if names.ndim != 1 or len(set(names.tolist())) != len(names) or bank["summaries"].shape[1] != len(names):
            raise ValueError("Invalid summary schema")
    if not np.array_equal(train["summary_names"], heldout["summary_names"]):
        raise ValueError("Training and held-out summary schemas differ")
    validate_groups(train["group_id"], heldout["group_id"])
    abc = RejectionABC(train["summaries"], train["model"], json.loads(args.model_prior.read_text()))
    diagnostics = heldout_diagnostics(abc, heldout["summaries"], heldout["model"], args.fraction)
    probabilities = diagnostics.pop("probabilities")
    nearest = diagnostics.pop("nearest_distances")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "diagnostics.json", diagnostics)
    np.savez_compressed(args.out / "heldout_predictions.npz", model_support=probabilities,
                        model=heldout["model"], group_id=heldout["group_id"], nearest_distance=nearest)
    if args.observed:
        obs = np.load(args.observed, allow_pickle=False)
        if not np.array_equal(obs["summary_names"], train["summary_names"]):
            raise ValueError("Observed summary schema differs")
        result = abc.infer(obs["summaries"], args.fraction)
        indices, weights, distances = [result.pop(key) for key in ["indices", "weights", "distances"]]
        result["model_support"] = dict(zip(abc.models, result["model_support"].tolist()))
        result["interpretation"] = "Finite-tolerance, summary-conditional ABC support; not biological validation"
        write_json(args.out / "observed_support.json", result)
        # Save the parameter rows by index; condition and renormalize by model
        # for density estimation. Do not mix inactive parameters across models.
        np.savez_compressed(args.out / "accepted.npz", row=indices, group_id=train["group_id"][indices],
                            model=train["model"][indices], weight=weights, distance=distances)
    write_json(args.out / "provenance.json", dict(
        status="unvalidated_reference_implementation", fraction=args.fraction,
        covariance_shrinkage=.1, models=abc.models, model_prior=abc.prior.tolist(),
        source_sha256=digest(__file__),
        inputs={str(p): digest(p) for p in [args.train, args.heldout, args.model_prior] + ([args.observed] if args.observed else [])}))


if __name__ == "__main__":
    main()
