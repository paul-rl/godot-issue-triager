import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    precision_score, recall_score
)


class TfidfOvrMultilabelBaseline:
    """
    TF-IDF + One-vs-Rest Logistic Regression for multi-label classification.

    - fit(train_df)
    - predict_proba(df)
    - tune thresholds on validation
    - predict with global/per-label thresholds + abstain
    - evaluate metrics
    """

    def __init__(
        self,
        *,
        class_names: list[str],
        tfidf_params: dict | None = None,
        lr_params: dict | None = None,
        threshold_policy: str = "per_label",  # "global" or "per_label"
        global_threshold: float = 0.5,
    ):
        self.class_names = list(class_names)

        self.tfidf = TfidfVectorizer(**(tfidf_params or {
            "lowercase": True,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
        }))

        base_lr = LogisticRegression(**(lr_params or {
            "max_iter": 2000,
            "class_weight": "balanced"
        }))
        self.clf = OneVsRestClassifier(base_lr)

        self.mlb = MultiLabelBinarizer(classes=self.class_names)

        # Thresholding state
        self.threshold_policy = threshold_policy
        self.global_threshold = float(global_threshold)
        self.per_label_thresholds: np.ndarray | None = None  # shape (K,)

        self._is_fit = False

    # -------------------------
    # Data helpers
    # -------------------------
    @staticmethod
    def _get_text_series(df: pd.DataFrame, text_col: str) -> pd.Series:
        return df[text_col].fillna("").astype(str)

    def _get_Y(self, df: pd.DataFrame, label_col: str, *, fit: bool = False) -> np.ndarray:
        y = self._normalize_labels(df[label_col])  # <- critical
        if fit:
            return self.mlb.fit_transform(y)
        return self.mlb.transform(y)

    # -------------------------
    # Normalize
    # -------------------------
    def _normalize_labels(self, series: pd.Series) -> pd.Series:
        """
        Make every row a list of labels:
          - NaN -> []
          - list/tuple/set -> list(...)
          - scalar (e.g. "bug") -> ["bug"]
        """
        def norm(x):
            # handle already-multilabel FIRST (avoid pd.isna(list) weirdness)
            if isinstance(x, (list, tuple, set)):
                return [str(v) for v in x if not pd.isna(v)]

            # now handle missing
            if x is None or pd.isna(x):
                return []

            # scalar -> wrap
            return [str(x)]

        return series.apply(norm)

    # -------------------------
    # Fit / Predict
    # -------------------------
    def fit(self, train_df: pd.DataFrame, *, text_col: str = "text_clean", label_col: str = "topic"):
        X_train = self.tfidf.fit_transform(self._get_text_series(train_df, text_col))
        Y_train = self._get_Y(train_df, label_col, fit=True)
        self.clf.fit(X_train, Y_train)
        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame, *, text_col: str = "text_clean"):
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        return self.tfidf.transform(self._get_text_series(df, text_col))

    def predict_proba(self, df: pd.DataFrame, *, text_col: str = "text_clean") -> np.ndarray:
        X = self.transform(df, text_col=text_col)
        return self.clf.predict_proba(X)  # (N, K)

    # -------------------------
    # Thresholding / Abstain
    # -------------------------
    def predict_with_thresholds(
        self,
        proba: np.ndarray,
        *,
        thresholds: np.ndarray | None = None,
        global_threshold: float | None = None,
    ):
        """
        Returns:
          Y_pred: (N, K) multi-hot
          abstain: (N,) bool
          pred_label_lists: list[list[str]]
        """
        if thresholds is not None:
            thr = thresholds
        else:
            thr = self.per_label_thresholds

        if thr is not None:
            Y_pred = (proba >= thr).astype(int)
        else:
            t = float(global_threshold if global_threshold is not None else self.global_threshold)
            Y_pred = (proba >= t).astype(int)

        abstain = (Y_pred.sum(axis=1) == 0)

        pred_label_lists = [
            [self.class_names[j] for j in np.where(Y_pred[i] == 1)[0]]
            for i in range(Y_pred.shape[0])
        ]
        return Y_pred, abstain, pred_label_lists

    # -------------------------
    # Tuning thresholds
    # -------------------------
    def tune_global_threshold(
        self,
        val_df: pd.DataFrame,
        *,
        text_col: str = "text_clean",
        label_col: str = "topic",
        grid: np.ndarray | None = None,
        select_by: str = "micro",  # "micro" or "macro"
    ):
        if grid is None:
            grid = np.linspace(0.05, 0.95, 30)

        proba = self.predict_proba(val_df, text_col=text_col)
        Y_val = self._get_Y(val_df, label_col)

        best_t, best_score = 0.5, -1.0
        for t in grid:
            Yp = (proba >= t).astype(int)
            score = f1_score(Y_val, Yp, average=select_by, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)

        self.global_threshold = best_t
        self.threshold_policy = "global"
        self.per_label_thresholds = None
        return {"best_threshold": best_t, f"best_{select_by}_f1": float(best_score)}

    def tune_per_label_thresholds_max_f1(
        self,
        val_df: pd.DataFrame,
        *,
        text_col: str = "text_clean",
        label_col: str = "topic",
        grid: np.ndarray | None = None,
    ):
        if grid is None:
            grid = np.linspace(0.05, 0.95, 19)

        proba = self.predict_proba(val_df, text_col=text_col)
        Y_val = self._get_Y(val_df, label_col)

        K = proba.shape[1]
        thresholds = np.full(K, 0.5, dtype=float)
        best_f1s = np.zeros(K, dtype=float)

        for j in range(K):
            best_t, best_f1 = 0.5, -1.0
            yj = Y_val[:, j]
            pj = proba[:, j]
            for t in grid:
                pred_j = (pj >= t).astype(int)
                f1 = f1_score(yj, pred_j, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds[j] = float(best_t)
            best_f1s[j] = float(best_f1)

        self.per_label_thresholds = thresholds
        self.threshold_policy = "per_label"
        return thresholds, best_f1s

    def tune_per_label_thresholds_min_precision(
        self,
        val_df,
        *,
        text_col="text_clean",
        label_col="topic",
        min_precision=0.70,
        grid=None,
        fallback="max_f1",   # "max_f1" | "max_precision" | "fixed"
        fixed_fallback_t=0.5,
        min_positives=20,
    ):
        if grid is None:
            grid = np.linspace(0.05, 0.95, 19)

        proba = self.predict_proba(val_df, text_col=text_col)
        Y_val = self._get_Y(val_df, label_col)

        N, K = proba.shape
        thresholds = np.full(K, fixed_fallback_t, dtype=float)
        summary = []

        for j in range(K):
            y_true = Y_val[:, j].astype(int)
            p = proba[:, j]
            pos = int(y_true.sum())

            def eval_at(t):
                y_pred = (p >= t).astype(int)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1v = f1_score(y_true, y_pred, zero_division=0)
                return prec, rec, f1v

            # low support -> fallback
            if pos < min_positives:
                best_t = fixed_fallback_t
                if fallback == "max_precision":
                    best = (-1.0, -1.0, best_t)  # (prec, rec, t)
                    for t in grid:
                        prec, rec, _ = eval_at(t)
                        if (prec > best[0]) or (prec == best[0] and rec > best[1]):
                            best = (prec, rec, t)
                    best_t = best[2]
                elif fallback == "max_f1":
                    best = (-1.0, best_t)  # (f1, t)
                    for t in grid:
                        _, _, f1v = eval_at(t)
                        if f1v > best[0]:
                            best = (f1v, t)
                    best_t = best[1]

                thresholds[j] = float(best_t)
                prec, rec, f1v = eval_at(best_t)
                summary.append({
                    "label": self.class_names[j],
                    "positives_val": pos,
                    "threshold": float(best_t),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1v),
                    "met_min_precision": bool(prec >= min_precision),
                    "note": f"fallback({fallback}) low_support<{min_positives}",
                })
                continue

            # thresholds meeting precision constraint
            candidates = []
            for t in grid:
                prec, rec, f1v = eval_at(t)
                if prec >= min_precision:
                    candidates.append((rec, f1v, prec, t))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                t_star = candidates[0][3]
                note = "min_precision_met"
            else:
                # fallback
                t_star = fixed_fallback_t
                if fallback == "max_precision":
                    best = (-1.0, -1.0, t_star)
                    for t in grid:
                        prec, rec, _ = eval_at(t)
                        if (prec > best[0]) or (prec == best[0] and rec > best[1]):
                            best = (prec, rec, t)
                    t_star = best[2]
                elif fallback == "max_f1":
                    best = (-1.0, t_star)
                    for t in grid:
                        _, _, f1v = eval_at(t)
                        if f1v > best[0]:
                            best = (f1v, t)
                    t_star = best[1]
                note = f"fallback({fallback}) cannot_meet_min_precision"

            thresholds[j] = float(t_star)
            prec, rec, f1v = eval_at(t_star)
            summary.append({
                "label": self.class_names[j],
                "positives_val": pos,
                "threshold": float(t_star),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1v),
                "met_min_precision": bool(prec >= min_precision),
                "note": note,
            })

        self.per_label_thresholds = thresholds
        self.threshold_policy = "per_label"
        return thresholds, summary

    # -------------------------
    # Evaluation
    # -------------------------
    def coverage_curve_from_proba(
        self,
        proba: np.ndarray,
        Y_true: np.ndarray,
        *,
        grid: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Build a coverage–accuracy curve by sweeping a SINGLE global threshold t
        (abstain = predicts zero labels).

        Returns a list of dict points: threshold, coverage, avg_pred_labels,
        plus metrics on:
          - all samples (abstentions count as empty predictions)
          - covered samples only (selective / conditional accuracy)
        """
        if grid is None:
            grid = np.linspace(0.05, 0.95, 19)

        curve = []
        K = proba.shape[1]
        for t in grid:
            # Force a global sweep even if per-label thresholds exist:
            # pass a (K,) vector with the same threshold for every label.
            thr_vec = np.full(K, float(t), dtype=float)
            Y_pred, abstain, _ = self.predict_with_thresholds(proba, thresholds=thr_vec)

            coverage = float((~abstain).mean())
            avg_pred_labels = float(Y_pred.sum(axis=1).mean())

            # All-samples metrics (abstain treated as empty prediction)
            micro_all = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
            macro_all = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
            ham_all = float(hamming_loss(Y_true, Y_pred))

            # Covered-only metrics (selective / conditional)
            mask = ~abstain
            if mask.any():
                micro_cov = float(f1_score(Y_true[mask], Y_pred[mask], average="micro", zero_division=0))
                macro_cov = float(f1_score(Y_true[mask], Y_pred[mask], average="macro", zero_division=0))
                ham_cov = float(hamming_loss(Y_true[mask], Y_pred[mask]))
            else:
                micro_cov = macro_cov = 0.0
                ham_cov = 0.0

            curve.append({
                "threshold": float(t),
                "coverage": coverage,
                "avg_pred_labels": avg_pred_labels,
                "micro_f1_all": micro_all,
                "macro_f1_all": macro_all,
                "hamming_loss_all": ham_all,
                "micro_f1_covered": micro_cov,
                "macro_f1_covered": macro_cov,
                "hamming_loss_covered": ham_cov,
            })

        # Nice for plotting later (coverage high -> low)
        curve.sort(key=lambda d: d["coverage"], reverse=True)
        return curve

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = "text_clean",
        label_col: str = "topic",
        use_policy: str | None = None,  # None -> use self.threshold_policy
        global_threshold: float | None = None,
        thresholds: np.ndarray | None = None,
        with_report: bool = True,
    ):
        proba = self.predict_proba(df, text_col=text_col)
        Y_true = self._get_Y(df, label_col)

        policy = use_policy or self.threshold_policy
        if policy == "per_label":
            Y_pred, abstain, pred_lists = self.predict_with_thresholds(proba, thresholds=thresholds)
        else:
            Y_pred, abstain, pred_lists = self.predict_with_thresholds(
                proba, thresholds=None, global_threshold=(global_threshold if global_threshold is not None else self.global_threshold)
            )

        coverage = float((~abstain).mean())
        avg_pred_labels = float(Y_pred.sum(axis=1).mean())

        metrics_all = {
            "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
            "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
            "coverage": coverage,
            "avg_pred_labels": avg_pred_labels,
        }

        # metrics @ coverage (exclude abstained)
        mask = ~abstain
        metrics_cov = None
        if mask.any():
            metrics_cov = {
                "micro_f1": float(f1_score(Y_true[mask], Y_pred[mask], average="micro", zero_division=0)),
                "macro_f1": float(f1_score(Y_true[mask], Y_pred[mask], average="macro", zero_division=0)),
                "hamming_loss": float(hamming_loss(Y_true[mask], Y_pred[mask])),
                "coverage": coverage,
            }

        report = None
        if with_report:
            report = classification_report(Y_true, Y_pred, target_names=self.class_names, zero_division=0)

        return {
            "metrics_all": metrics_all,
            "metrics_at_coverage": metrics_cov,
            "abstain": abstain,
            "pred_label_lists": pred_lists,
            "report": report,
            "proba": proba,
            "Y_true": Y_true,
            "Y_pred": Y_pred,
        }

    # -------------------------
    # Ranking metrics (Hit@k, Recall@k)
    # -------------------------
    @staticmethod
    def hit_rate_at_k(proba, Y_true, k: int = 5) -> float:
        proba = np.asarray(proba)
        Y_true = np.asarray(Y_true)

        n_classes = proba.shape[1]
        if n_classes == 0:
            return 0.0

        k_eff = min(k, n_classes)
        if k_eff <= 0:
            return 0.0

        # top-k indices per row
        topk_idx = np.argpartition(-proba, kth=k_eff - 1, axis=1)[:, :k_eff]

        # hit if any true label is in top-k
        rows = np.arange(Y_true.shape[0])[:, None]
        hits = (Y_true[rows, topk_idx] > 0).any(axis=1)

        # OPTIONAL: if some rows have no true labels, you can exclude them:
        # has_label = Y_true.sum(axis=1) > 0
        # if has_label.any():
        #     return hits[has_label].mean()
        # return 0.0

        return hits.mean()

    @staticmethod
    def recall_at_k(proba, Y_true, k: int = 5, average: str = "micro") -> float:
        proba = np.asarray(proba)
        Y_true = np.asarray(Y_true)

        n_classes = proba.shape[1]
        k_eff = min(k, n_classes)
        if k_eff <= 0:
            return 0.0

        topk_idx = np.argpartition(-proba, kth=k_eff - 1, axis=1)[:, :k_eff]

        rows = np.arange(Y_true.shape[0])[:, None]
        hits = (Y_true[rows, topk_idx] > 0)  # (N, k_eff) bool

        true_counts = Y_true.sum(axis=1)     # (N,)
        hit_counts = hits.sum(axis=1)        # (N,)

        # per-example recall = (# true labels retrieved in top-k) / (# true labels)
        denom = np.maximum(true_counts, 1)

        per_example_recall = hit_counts / denom

        if average == "micro":
            # micro: total hits / total true labels (excluding rows with 0 true labels)
            total_true = true_counts.sum()
            if total_true == 0:
                return 0.0
            return hit_counts.sum() / total_true

        if average == "macro":
            # macro: mean over examples that have at least one true label
            mask = true_counts > 0
            if not mask.any():
                return 0.0
            return per_example_recall[mask].mean()

        raise ValueError("average must be 'micro' or 'macro'")