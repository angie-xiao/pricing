# simplifying model
import numpy as np
from typing import Optional
from pygam import ExpectileGAM, f, te


class ParamSearchCV:
    def _fit_price_curve_with_anchors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        *,
        expectile: float = 0.5,
        n_splines: int = 16,
        lam: float = 500.0,
        max_iter: int = 6000,
        tol: float = 2e-4,
    ):
        """
        Fits GAM with all features
        """
        if ExpectileGAM is None:
            raise RuntimeError("pygam is required for fit_price_curve_with_anchors")

        # Preprocess features - fit scalers on input data
        X = self._preprocess_features(X, fit=True)

        X = np.asarray(X, float)
        y = np.asarray(y, float)
        p = X[:, 0]  # price still in first column

        # anchor construction
        # p is scaled already; compute edges in scaled space
        span = (p.max() - p.min()) or 1.0
        eps = 1e-3 * span

        median_disc = float(np.median(X[:, 1]))  # already scaled
        median_event = int(np.median(X[:, 2]))
        median_product = int(np.median(X[:, 3]))

        X_anchor = np.array(
            [
                [p.min() - eps, median_disc, median_event, median_product],
                [p.max() + eps, median_disc, median_event, median_product],
            ],
            dtype=float,
        )

        y_lo = (
            float(np.median(y[p <= np.quantile(p, 0.10)]))
            if np.any(p <= np.quantile(p, 0.10))
            else float(np.median(y))
        )
        y_hi = (
            float(np.median(y[p >= np.quantile(p, 0.90)]))
            if np.any(p >= np.quantile(p, 0.90))
            else float(np.median(y))
        )
        y_anchor = np.array([y_lo, y_hi], dtype=float)

        # Tiny anchor weights
        w_anchor = np.full(X_anchor.shape[0], 1e-3, dtype=float)
        if w is None:
            w = np.ones_like(y, dtype=float)
        w_aug = np.concatenate([w, w_anchor])

        X_aug = np.vstack([X, X_anchor])
        y_aug = np.concatenate([y, y_anchor])

        # Create proper TermList
        terms = te(0, 1, n_splines=[int(n_splines), 10]) + f(2) + f(3)

        gam = ExpectileGAM(
            terms,
            lam=float(lam),
            expectile=float(expectile),
            max_iter=int(max_iter),
            tol=float(tol),
        )
        gam.fit(X_aug, y_aug, weights=w_aug)

        return gam
