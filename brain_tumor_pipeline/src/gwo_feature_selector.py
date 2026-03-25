from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .config import FeatureSelectionConfig


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class WolfSolution:
    fitness: float
    accuracy: float
    feature_count: int
    feature_indices: Tuple[int, ...]
    iteration: int


class GreyWolfFeatureSelector:
    """
    Binary GWO over feature subsets.

    Representation:
    - Each wolf position is a continuous vector X in R^d.
    - Convert to probabilities via sigmoid(X).
    - Select a variable number of features k derived from the sum of probabilities,
      then take the top-k features by probability (deterministic, stable).
    """

    def __init__(
        self,
        *,
        cfg: FeatureSelectionConfig,
        total_features: int,
        random_state: int,
    ):
        self.cfg = cfg
        self.total_features = total_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.svm_model_cache: Dict[Tuple[int, bytes], float] = {}
        self.best_by_iter: List[WolfSolution] = []
        self.top_history: List[WolfSolution] = []

        self.alpha: Optional[WolfSolution] = None
        self.beta: Optional[WolfSolution] = None
        self.delta: Optional[WolfSolution] = None

    def _lambda_size(self, iter_idx: int) -> float:
        t = iter_idx / max(1, self.cfg.max_iterations - 1)
        # Increase the sparsity penalty over time to move from accuracy toward minimal subsets.
        return self.cfg.lambda_size_start + (self.cfg.lambda_size_end - self.cfg.lambda_size_start) * (t**2)

    def _subset_from_position(self, position: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        probs = sigmoid(position)
        expected_k = float(probs.sum())
        k = int(np.clip(int(np.round(expected_k)), self.cfg.min_features, self.cfg.max_features))
        k = max(1, min(k, self.total_features))

        # Deterministically keep top-k highest probabilities.
        if k == self.total_features:
            idxs = np.arange(self.total_features, dtype=np.int64)
        else:
            idxs_part = np.argpartition(probs, -k)[-k:]
            idxs = idxs_part[np.argsort(probs[idxs_part])[::-1]]
        return idxs, k, probs

    def _eval_subset_accuracy(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> float:
        key_bytes = feature_indices.astype(np.int32).tobytes()
        key = (len(feature_indices), key_bytes)
        cached = self.svm_model_cache.get(key)
        if cached is not None:
            return cached

        X_sub = X[:, feature_indices]
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.random_state)
        accs: List[float] = []
        for tr_idx, te_idx in skf.split(X_sub, y):
            X_tr, X_te = X_sub[tr_idx], X_sub[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            clf = SVC(
                kernel="rbf",
                C=self.cfg.svm_C,
                gamma=self.cfg.svm_gamma,
                probability=False,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            accs.append(float(accuracy_score(y_te, y_pred)))

        acc = float(np.mean(accs))
        self.svm_model_cache[key] = acc
        return acc

    def _evaluate_wolf(self, X: np.ndarray, y: np.ndarray, position: np.ndarray, iter_idx: int) -> WolfSolution:
        idxs, k, _probs = self._subset_from_position(position)
        acc = self._eval_subset_accuracy(X, y, idxs)

        lambda_size = self._lambda_size(iter_idx)
        size_ratio = k / max(1, self.total_features)
        fitness = acc - lambda_size * size_ratio
        return WolfSolution(
            fitness=float(fitness),
            accuracy=float(acc),
            feature_count=int(k),
            feature_indices=tuple(map(int, idxs.tolist())),
            iteration=int(iter_idx),
        )

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize continuous positions. We bias initial probabilities toward the feasible subset size range.
        """
        pop = self.cfg.population_size
        d = self.total_features

        # Start with expected subset size near the middle of [min_features, max_features].
        target_k = int(0.5 * (self.cfg.min_features + self.cfg.max_features))
        target_k = max(1, min(target_k, d))
        p0 = float(target_k / d)
        # Convert desired initial probabilities into logits ~ centered around logit(p0).
        logit = np.log(p0 + 1e-6) - np.log(1.0 - p0 + 1e-6)

        positions = self._rng.normal(loc=logit, scale=1.0, size=(pop, d)).astype(np.float32)
        return positions

    def _update_best(self, sol: WolfSolution) -> None:
        # Update alpha/beta/delta and maintain top-history.
        if self.alpha is None or sol.fitness > self.alpha.fitness:
            self.delta = self.beta
            self.beta = self.alpha
            self.alpha = sol
        elif self.beta is None or sol.fitness > self.beta.fitness:
            self.delta = self.beta
            self.beta = sol
        elif self.delta is None or sol.fitness > self.delta.fitness:
            self.delta = sol

        self.top_history.append(sol)
        # Keep top solutions (by fitness).
        self.top_history = sorted(self.top_history, key=lambda s: s.fitness, reverse=True)[: self.cfg.top_k_history]

    def _positions_from_feature_indices(self, idxs: Tuple[int, ...], *, d: int) -> np.ndarray:
        """
        Reconstruct a continuous position vector from a selected subset.
        """
        pos = self._rng.normal(0.0, 0.5, size=(d,)).astype(np.float32)
        pos[list(idxs)] += 3.0  # push these dimensions high => high sigmoid probs
        return pos

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        X = np.asarray(X)
        y = np.asarray(y)
        d = X.shape[1]
        if d != self.total_features:
            raise ValueError(f"total_features mismatch: cfg={self.total_features} X has d={d}")

        positions = self._initialize_population()

        best_fitness_prev: Optional[float] = None
        stagnation_counter = 0

        for iter_idx in range(self.cfg.max_iterations):
            # Evaluate all wolves.
            alpha = None
            beta = None
            delta = None
            iter_best: Optional[WolfSolution] = None

            for i in range(self.cfg.population_size):
                sol = self._evaluate_wolf(X, y, positions[i], iter_idx)

                if iter_best is None or sol.fitness > iter_best.fitness:
                    iter_best = sol
                if alpha is None or sol.fitness > alpha.fitness:
                    delta = beta
                    beta = alpha
                    alpha = sol
                elif beta is None or sol.fitness > beta.fitness:
                    delta = beta
                    beta = sol
                elif delta is None or sol.fitness > delta.fitness:
                    delta = sol

            assert iter_best is not None
            self.best_by_iter.append(iter_best)
            self.alpha, self.beta, self.delta = alpha, beta, delta
            self._update_best(iter_best)

            current_best = float(iter_best.fitness)
            if best_fitness_prev is not None:
                if current_best - best_fitness_prev <= self.cfg.min_accuracy_gain_tol:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
            best_fitness_prev = current_best

            # Adaptive convergence control: non-linear decay of "a".
            a = 2.0 * ((1.0 - iter_idx / max(1, self.cfg.max_iterations)) ** self.cfg.a_decay_power)

            # Periodic reinit (optional).
            do_reinit = self.cfg.reinitialize_every > 0 and (iter_idx > 0) and (iter_idx % self.cfg.reinitialize_every == 0)
            # Reinit on stagnation.
            if stagnation_counter >= self.cfg.stagnation_patience:
                do_reinit = True

            if do_reinit:
                # Pick from historically best wolves and reconstruct positions, then mutate slightly.
                history = self.top_history[: max(1, min(self.cfg.top_k_history, len(self.top_history)))]
                if not history:
                    history = []
                positions = np.empty_like(positions)
                for wi in range(self.cfg.population_size):
                    if history:
                        pick = history[int(self._rng.integers(0, len(history)))]
                        positions[wi] = self._positions_from_feature_indices(pick.feature_indices, d=d)
                    else:
                        positions[wi] = self._rng.normal(0.0, 1.0, size=(d,)).astype(np.float32)

                    # Mutation: randomly flip some dimensions by nudging logits.
                    mut_mask = self._rng.random(size=(d,)) < self.cfg.mutation_flip_prob
                    positions[wi][mut_mask] *= -1.0

                stagnation_counter = 0

            # GWO position update.
            # We update each wolf using the alpha/beta/delta positions.
            alpha_pos = alpha.feature_indices  # type: ignore[union-attr]

            # Convert stored alpha/beta/delta solutions into continuous positions approximately by reconstruction.
            alpha_vec = self._positions_from_feature_indices(alpha.feature_indices, d=d) if alpha is not None else positions[0]  # type: ignore[arg-type]
            beta_vec = self._positions_from_feature_indices(beta.feature_indices, d=d) if beta is not None else positions[0]  # type: ignore[arg-type]
            delta_vec = self._positions_from_feature_indices(delta.feature_indices, d=d) if delta is not None else positions[0]  # type: ignore[arg-type]

            # Update all wolves in-place.
            for i in range(self.cfg.population_size):
                for vec, _name in [(alpha_vec, "alpha"), (beta_vec, "beta"), (delta_vec, "delta")]:
                    pass  # for readability; no-op

                r1 = self._rng.random(size=d).astype(np.float32)
                r2 = self._rng.random(size=d).astype(np.float32)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_vec - positions[i])
                X1 = alpha_vec - A1 * D_alpha

                r1 = self._rng.random(size=d).astype(np.float32)
                r2 = self._rng.random(size=d).astype(np.float32)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_vec - positions[i])
                X2 = beta_vec - A2 * D_beta

                r1 = self._rng.random(size=d).astype(np.float32)
                r2 = self._rng.random(size=d).astype(np.float32)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_vec - positions[i])
                X3 = delta_vec - A3 * D_delta

                positions[i] = (X1 + X2 + X3) / 3.0

        # Choose optimal subset size automatically:
        # from best_by_iter, pick highest accuracy, then smallest feature_count.
        unique_by_mask: Dict[Tuple[int, ...], WolfSolution] = {}
        for sol in self.best_by_iter:
            unique_by_mask[sol.feature_indices] = sol

        ranked = sorted(
            unique_by_mask.values(),
            key=lambda s: (-s.accuracy, s.feature_count),
        )
        best = ranked[0] if ranked else self.best_by_iter[-1]

        return {
            "best_feature_indices": list(best.feature_indices),
            "best_feature_count": int(best.feature_count),
            "best_accuracy_cv": float(best.accuracy),
            "best_fitness": float(best.fitness),
            "best_by_iter": [
                {
                    "iter": int(s.iteration),
                    "fitness": float(s.fitness),
                    "accuracy": float(s.accuracy),
                    "feature_count": int(s.feature_count),
                }
                for s in self.best_by_iter
            ],
            "top_history": [
                {
                    "fitness": float(s.fitness),
                    "accuracy": float(s.accuracy),
                    "feature_count": int(s.feature_count),
                    "feature_indices": list(s.feature_indices),
                    "iter": int(s.iteration),
                }
                for s in sorted(unique_by_mask.values(), key=lambda s: s.fitness, reverse=True)[: self.cfg.top_k_history]
            ],
        }

