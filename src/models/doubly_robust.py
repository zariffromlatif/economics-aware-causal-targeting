"""Doubly Robust Learner (Family C) with Cross-Fitting.

Implements Augmented Inverse Propensity Weighting (AIPW) / DR-Learner.
Combines outcome regression and propensity weighting with K-fold cross-fitting
to achieve orthogonalized, double-machine-learning guarantees (Chernozhukov et al., 2018).
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb


class DoublyRobustLearner:
    """Doubly Robust (DR) Learner with K-fold cross-fitting.
    
    Transforms the causal estimation problem into pseudo-outcome regression:
    Gamma_i = mu_1(X_i) - mu_0(X_i) + [T_i * (Y_i - mu_1(X_i))] / e(X_i)
                                    - [(1 - T_i) * (Y_i - mu_0(X_i))] / (1 - e(X_i))
    tau(X) = argmin_f E[(Gamma - f(X))^2]
    """

    def __init__(
        self,
        n_splits: int = 5,
        final_regressor: Optional[BaseEstimator] = None,
        is_classification: bool = True,
        use_known_propensity: bool = True,
        known_propensity: float = 0.5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.is_classification = is_classification
        self.use_known_propensity = use_known_propensity
        self.known_propensity = known_propensity
        self.random_state = random_state

        if final_regressor is None:
            self.final_regressor = xgb.XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=random_state
            )
        else:
            self.final_regressor = clone(final_regressor)

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series) -> "DoublyRobustLearner":
        """Compute cross-fitted doubly robust pseudo-outcomes and fit final stage."""
        N = len(X)
        X_arr = X.values
        T_arr = T.values.astype(int)
        y_arr = y.values.astype(float)

        gamma = np.zeros(N)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X, T):
            X_tr, T_tr, y_tr = X_arr[train_idx], T_arr[train_idx], y_arr[train_idx]
            X_va, T_va, y_va = X_arr[val_idx], T_arr[val_idx], y_arr[val_idx]

            # Fit propensity model e(X)
            if self.use_known_propensity:
                e_hat = np.full(len(val_idx), self.known_propensity)
            else:
                prop_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                prop_model.fit(X_tr, T_tr)
                e_hat = prop_model.predict_proba(X_va)[:, 1]
                e_hat = np.clip(e_hat, 0.01, 0.99)

            # Fit outcome models
            mask_0 = (T_tr == 0)
            mask_1 = (T_tr == 1)

            if self.is_classification:
                m0 = LogisticRegression(max_iter=1000, random_state=self.random_state)
                m1 = LogisticRegression(max_iter=1000, random_state=self.random_state)
                m0.fit(X_tr[mask_0], y_tr[mask_0])
                m1.fit(X_tr[mask_1], y_tr[mask_1])
                mu0 = m0.predict_proba(X_va)[:, 1]
                mu1 = m1.predict_proba(X_va)[:, 1]
            else:
                m0 = Ridge(alpha=1.0)
                m1 = Ridge(alpha=1.0)
                m0.fit(X_tr[mask_0], y_tr[mask_0])
                m1.fit(X_tr[mask_1], y_tr[mask_1])
                mu0 = m0.predict(X_va)
                mu1 = m1.predict(X_va)

            # AIPW pseudo-outcome formula
            gamma_val = (
                (mu1 - mu0)
                + (T_va * (y_va - mu1)) / e_hat
                - ((1.0 - T_va) * (y_va - mu0)) / (1.0 - e_hat)
            )
            gamma[val_idx] = gamma_val

        # Final stage: Regress AIPW score Gamma on features X
        self.final_regressor.fit(X, gamma)
        self.train_gamma_ = gamma
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE) tau(X)."""
        return self.final_regressor.predict(X)
