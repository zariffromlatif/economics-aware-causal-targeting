"""Incremental Revenue Modeling.

Addresses the zero-inflated nature of e-commerce spend data.
Implements:
1. Direct Continuous CATE Learner on Spend (T-Learner, DR-Learner).
2. Two-Stage Hurdle Uplift Model:
   Decomposes expected revenue into conversion uplift and conditional basket spend:
   E[Spend | X, T] = P(Conv=1 | X, T) * E[Spend | Conv=1, X, T]
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

from src.models.meta_learners import TLearner, XLearner
from src.models.doubly_robust import DoublyRobustLearner


class HurdleRevenueUplift:
    """Two-stage Hurdle model for incremental revenue.
    
    Stage 1: Model conversion probability uplift P(Conv=1 | X, T).
    Stage 2: Model expected spend conditional on conversion E[Spend | Conv=1, X, T].
    Delta R(X) = mu_1^C(X) * mu_1^S(X) - mu_0^C(X) * mu_0^S(X)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Conversion stage (T-Learner for conversion probability)
        self.conv_learner = TLearner(is_classification=True)
        # Conditional spend stage (regressor fitted on converters only)
        self.spend_model_0 = Ridge(alpha=10.0)
        self.spend_model_1 = Ridge(alpha=10.0)

    def fit(self, X: pd.DataFrame, T: pd.Series, y_conv: pd.Series, y_spend: pd.Series) -> "HurdleRevenueUplift":
        """Fit conversion stage on full data, spend stage on converters."""
        # Stage 1: Conversion models
        self.conv_learner.fit(X, T, y_conv)

        # Stage 2: Conditional spend
        conv_mask = (y_conv.values == 1)
        X_conv = X.loc[conv_mask]
        T_conv = T.loc[conv_mask].values
        y_spend_conv = y_spend.loc[conv_mask].values

        mask_c0 = (T_conv == 0)
        mask_c1 = (T_conv == 1)

        # If converter counts in an arm are small, fall back to overall converter mean
        if mask_c0.sum() > 5:
            self.spend_model_0.fit(X_conv.loc[mask_c0], y_spend_conv[mask_c0])
        else:
            self.spend_model_0.fit(X_conv, y_spend_conv)

        if mask_c1.sum() > 5:
            self.spend_model_1.fit(X_conv.loc[mask_c1], y_spend_conv[mask_c1])
        else:
            self.spend_model_1.fit(X_conv, y_spend_conv)

        return self

    def predict_incremental_revenue(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate incremental spend Delta R(X)."""
        p1 = self.conv_learner.predict_mu1(X)
        p0 = self.conv_learner.predict_mu0(X)

        s1 = np.maximum(0.0, self.spend_model_1.predict(X))
        s0 = np.maximum(0.0, self.spend_model_0.predict(X))

        # Expected revenue under T=1 minus expected revenue under T=0
        expected_r1 = p1 * s1
        expected_r0 = p0 * s0
        return expected_r1 - expected_r0


class DirectRevenueUplift:
    """Direct continuous causal learner on spend."""

    def __init__(self, method: str = "t_learner", random_state: int = 42):
        self.method = method.lower()
        self.random_state = random_state
        if self.method == "t_learner":
            self.learner = TLearner(is_classification=False)
        elif self.method == "dr_learner":
            self.learner = DoublyRobustLearner(is_classification=False, random_state=random_state)
        elif self.method == "x_learner":
            self.learner = XLearner(is_classification=False)
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(self, X: pd.DataFrame, T: pd.Series, y_spend: pd.Series) -> "DirectRevenueUplift":
        """Fit direct continuous learner on spend."""
        self.learner.fit(X, T, y_spend)
        return self

    def predict_incremental_revenue(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate incremental spend Delta R(X)."""
        return self.learner.predict_cate(X)
