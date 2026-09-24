"""Conventional Response Prediction Baselines (Family A).

Implements:
- Model A1: Logistic Regression predicting P(Y=1 | X)
- Model A2: Gradient Boosted Trees (XGBoost) predicting P(Y=1 | X)

Evaluates predictive performance (ROC-AUC, PR-AUC, Brier score, Log Loss)
on held-out test data. Demonstrates the conventional marketing analytics baseline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
import xgboost as xgb


@dataclass
class ResponseModelResults:
    """Evaluation metrics for response prediction models."""
    model_name: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss_val: float
    train_score: float


class ResponseBaseline:
    """Wrapper for training and scoring conventional response models."""

    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=random_state,
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResponseBaseline":
        """Fit response model on customer features X and outcome y."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict response probability P(Y=1 | X)."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ResponseModelResults:
        """Evaluate model performance on test set."""
        preds = self.predict_proba(X_test)
        preds = np.clip(preds, 1e-15, 1 - 1e-15)

        return ResponseModelResults(
            model_name=self.model_type,
            roc_auc=float(roc_auc_score(y_test, preds)),
            pr_auc=float(average_precision_score(y_test, preds)),
            brier_score=float(brier_score_loss(y_test, preds)),
            log_loss_val=float(log_loss(y_test, preds)),
            train_score=0.0,
        )


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outcome_name: str = "conversion",
) -> Tuple[Dict[str, ResponseBaseline], pd.DataFrame]:
    """Train both Logistic Regression and XGBoost response models and summarize results."""
    models = {}
    records = []

    for m_type in ["logistic", "xgboost"]:
        baseline = ResponseBaseline(model_type=m_type)
        baseline.fit(X_train, y_train)
        eval_res = baseline.evaluate(X_test, y_test)
        models[m_type] = baseline
        records.append({
            "Outcome": outcome_name,
            "Model": f"Model A{1 if m_type == 'logistic' else 2} ({m_type.capitalize()})",
            "ROC-AUC": eval_res.roc_auc,
            "PR-AUC": eval_res.pr_auc,
            "Brier Score": eval_res.brier_score,
            "Log Loss": eval_res.log_loss_val,
        })

    summary_df = pd.DataFrame(records)
    return models, summary_df


if __name__ == "__main__":
    from src.data.loader import prepare_hillstrom_splits
    splits = prepare_hillstrom_splits()

    print("Training Response Models on Conversion...")
    models, summary = train_and_evaluate_baselines(
        splits.X_train, splits.y_conv_train,
        splits.X_test, splits.y_conv_test,
        outcome_name="Conversion (Y^C)"
    )
    print(summary.to_markdown(index=False))
