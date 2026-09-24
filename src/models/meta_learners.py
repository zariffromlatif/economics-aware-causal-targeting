"""Causal Uplift Meta-Learners (Family B).

Implements canonical causal meta-learning estimators:
- Model B1: S-Learner (Single model with treatment indicator)
- Model B2: T-Learner (Twin models for treatment and control)
- Model B3: X-Learner (Künzel et al., 2019 - cross-imputed counterfactuals)

Supports both classification (conversion uplift) and continuous regression (spend/revenue uplift).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb


class SLearner:
    """S-Learner: Single model incorporating treatment as a feature.
    
    tau(X) = mu(X, 1) - mu(X, 0)
    """

    def __init__(self, base_estimator: Optional[BaseEstimator] = None, is_classification: bool = True):
        self.is_classification = is_classification
        if base_estimator is None:
            if is_classification:
                self.base_estimator = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric="logloss"
                )
            else:
                self.base_estimator = xgb.XGBRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
        else:
            self.base_estimator = clone(base_estimator)

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series) -> "SLearner":
        """Fit single model on concatenated features and treatment indicator."""
        X_with_t = X.copy()
        X_with_t["treatment_indicator"] = T.values
        self.base_estimator.fit(X_with_t, y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE) tau(X)."""
        X_t1 = X.copy()
        X_t1["treatment_indicator"] = 1.0

        X_t0 = X.copy()
        X_t0["treatment_indicator"] = 0.0

        if self.is_classification:
            mu1 = self.base_estimator.predict_proba(X_t1)[:, 1]
            mu0 = self.base_estimator.predict_proba(X_t0)[:, 1]
        else:
            mu1 = self.base_estimator.predict(X_t1)
            mu0 = self.base_estimator.predict(X_t0)

        return mu1 - mu0


class TLearner:
    """T-Learner: Separate models for treatment and control subpopulations.
    
    tau(X) = mu_1(X) - mu_0(X)
    """

    def __init__(self, base_estimator: Optional[BaseEstimator] = None, is_classification: bool = True):
        self.is_classification = is_classification
        if base_estimator is None:
            if is_classification:
                self.model_0 = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric="logloss"
                )
                self.model_1 = clone(self.model_0)
            else:
                self.model_0 = xgb.XGBRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                self.model_1 = clone(self.model_0)
        else:
            self.model_0 = clone(base_estimator)
            self.model_1 = clone(base_estimator)

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series) -> "TLearner":
        """Fit model_0 on control (T=0) and model_1 on treated (T=1)."""
        mask_0 = (T.values == 0)
        mask_1 = (T.values == 1)

        self.model_0.fit(X.loc[mask_0], y.loc[mask_0])
        self.model_1.fit(X.loc[mask_1], y.loc[mask_1])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate CATE tau(X) = mu_1(X) - mu_0(X)."""
        if self.is_classification:
            mu1 = self.model_1.predict_proba(X)[:, 1]
            mu0 = self.model_0.predict_proba(X)[:, 1]
        else:
            mu1 = self.model_1.predict(X)
            mu0 = self.model_0.predict(X)
        return mu1 - mu0

    def predict_mu0(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected control potential outcome."""
        if self.is_classification:
            return self.model_0.predict_proba(X)[:, 1]
        return self.model_0.predict(X)

    def predict_mu1(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected treated potential outcome."""
        if self.is_classification:
            return self.model_1.predict_proba(X)[:, 1]
        return self.model_1.predict(X)


class XLearner:
    """X-Learner (Künzel et al., 2019): Cross-imputed counterfactual residuals.
    
    Particularly effective in settings with unbalanced treatment assignment
    or heterogeneous response densities.
    """

    def __init__(self, base_estimator: Optional[BaseEstimator] = None, is_classification: bool = True):
        self.is_classification = is_classification
        self.t_learner = TLearner(base_estimator=base_estimator, is_classification=is_classification)
        
        # Second stage residual estimators (always regressors since imputed residuals are continuous)
        self.effect_model_0 = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        self.effect_model_1 = clone(self.effect_model_0)
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series) -> "XLearner":
        """Fit stage 1 T-Learner, compute imputed counterfactuals, fit stage 2."""
        # Stage 1: Fit T-Learner
        self.t_learner.fit(X, T, y)

        # Propensity model e(X) = P(T=1 | X)
        self.propensity_model.fit(X, T)

        mask_0 = (T.values == 0)
        mask_1 = (T.values == 1)

        X_0, y_0 = X.loc[mask_0], y.loc[mask_0].values
        X_1, y_1 = X.loc[mask_1], y.loc[mask_1].values

        # Impute counterfactuals:
        # D_1 = Y_1 - mu_0(X_1)
        # D_0 = mu_1(X_0) - Y_0
        mu0_on_1 = self.t_learner.predict_mu0(X_1)
        mu1_on_0 = self.t_learner.predict_mu1(X_0)

        D_1 = y_1 - mu0_on_1
        D_0 = mu1_on_0 - y_0

        # Stage 2: Fit effect models on imputed effects
        self.effect_model_1.fit(X_1, D_1)
        self.effect_model_0.fit(X_0, D_0)

        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Combine stage 2 effect models using estimated propensity score."""
        e_hat = self.propensity_model.predict_proba(X)[:, 1]
        tau_1 = self.effect_model_1.predict(X)
        tau_0 = self.effect_model_0.predict(X)

        # tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X)
        return e_hat * tau_0 + (1.0 - e_hat) * tau_1
