"""Data preprocessing, feature encoding, and leakage-controlled splitting.

Prepares the Hillstrom dataset for causal estimation and economic policy evaluation.
Ensures zero data leakage from post-treatment outcomes into feature matrices.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    """Container for train, validation, and test datasets."""
    X_train: pd.DataFrame
    T_train: pd.Series
    y_spend_train: pd.Series
    y_conv_train: pd.Series
    y_visit_train: pd.Series

    X_val: pd.DataFrame
    T_val: pd.Series
    y_spend_val: pd.Series
    y_conv_val: pd.Series
    y_visit_val: pd.Series

    X_test: pd.DataFrame
    T_test: pd.Series
    y_spend_test: pd.Series
    y_conv_test: pd.Series
    y_visit_test: pd.Series

    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame


PRE_TREATMENT_NUMERIC = ["recency", "history"]
PRE_TREATMENT_BINARY = ["mens", "womens", "newbie"]
PRE_TREATMENT_CATEGORICAL = ["zip_code", "channel"]
# Note: history_segment is a deterministic discretization of `history`.
# We drop it by default to avoid multicollinearity, or keep one-hot encoded if desired.

POST_TREATMENT_OUTCOMES = ["spend", "conversion", "visit"]
TREATMENT_COL = "segment"


def audit_feature_leakage(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Verify that no post-treatment or outcome columns are present in features."""
    leaked = set(feature_cols).intersection(POST_TREATMENT_OUTCOMES)
    if leaked:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! Outcome columns in feature set: {leaked}")
    if TREATMENT_COL in feature_cols:
        raise ValueError("CRITICAL LEAKAGE DETECTED! Treatment column in feature set.")


def encode_features(
    df: pd.DataFrame,
    categorical_cols: List[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Perform one-hot encoding for categorical pre-treatment features.

    Keeps numerical and binary indicators intact.
    """
    cat_cols = categorical_cols or PRE_TREATMENT_CATEGORICAL
    base_features = PRE_TREATMENT_NUMERIC + PRE_TREATMENT_BINARY

    # Audit before encoding
    audit_feature_leakage(df, base_features + cat_cols)

    # Encode categoricals with dummy variables
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=drop_first, dtype=float)
    X = pd.concat([df[base_features].astype(float), df_encoded], axis=1)

    return X


def prepare_hillstrom_splits(
    csv_path: str = "data/raw/hillstrom.csv",
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    treatment_mapping: Dict[str, int] | None = None,
) -> DatasetSplits:
    """Load raw Hillstrom data, audit features, encode, and split into train/val/test.

    Splitting is stratified by treatment assignment to guarantee balance across splits.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    path = Path(csv_path)
    if not path.exists():
        print(f"[INFO] Dataset not found at {csv_path}. Auto-downloading...")
        from src.data.download import download_hillstrom
        download_hillstrom(target_path=str(path))

    raw_df = pd.read_csv(path)

    # Default treatment mapping:
    # 0 = No E-Mail (Control), 1 = Mens E-Mail, 2 = Womens E-Mail
    default_mapping = {
        "No E-Mail": 0,
        "Mens E-Mail": 1,
        "Womens E-Mail": 2,
    }
    mapping = treatment_mapping or default_mapping

    raw_df["treatment_arm"] = raw_df[TREATMENT_COL].map(mapping)
    if raw_df["treatment_arm"].isnull().any():
        raise ValueError("Unrecognized values in treatment segment column.")

    # Generate feature matrix
    X_full = encode_features(raw_df)
    T_full = raw_df["treatment_arm"]
    y_spend = raw_df["spend"]
    y_conv = raw_df["conversion"]
    y_visit = raw_df["visit"]

    # Stratified split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        raw_df.index,
        test_size=temp_ratio,
        random_state=seed,
        stratify=T_full,
    )

    # Stratified split: Val vs Test
    val_share_of_temp = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=T_full.loc[temp_idx],
    )

    return DatasetSplits(
        X_train=X_full.loc[train_idx].copy().reset_index(drop=True),
        T_train=T_full.loc[train_idx].copy().reset_index(drop=True),
        y_spend_train=y_spend.loc[train_idx].copy().reset_index(drop=True),
        y_conv_train=y_conv.loc[train_idx].copy().reset_index(drop=True),
        y_visit_train=y_visit.loc[train_idx].copy().reset_index(drop=True),

        X_val=X_full.loc[val_idx].copy().reset_index(drop=True),
        T_val=T_full.loc[val_idx].copy().reset_index(drop=True),
        y_spend_val=y_spend.loc[val_idx].copy().reset_index(drop=True),
        y_conv_val=y_conv.loc[val_idx].copy().reset_index(drop=True),
        y_visit_val=y_visit.loc[val_idx].copy().reset_index(drop=True),

        X_test=X_full.loc[test_idx].copy().reset_index(drop=True),
        T_test=T_full.loc[test_idx].copy().reset_index(drop=True),
        y_spend_test=y_spend.loc[test_idx].copy().reset_index(drop=True),
        y_conv_test=y_conv.loc[test_idx].copy().reset_index(drop=True),
        y_visit_test=y_visit.loc[test_idx].copy().reset_index(drop=True),

        raw_train=raw_df.loc[train_idx].copy().reset_index(drop=True),
        raw_val=raw_df.loc[val_idx].copy().reset_index(drop=True),
        raw_test=raw_df.loc[test_idx].copy().reset_index(drop=True),
    )


if __name__ == "__main__":
    splits = prepare_hillstrom_splits()
    print(f"Data successfully split:")
    print(f"  Train: {len(splits.X_train):,} samples ({splits.X_train.shape[1]} features)")
    print(f"  Val:   {len(splits.X_val):,} samples")
    print(f"  Test:  {len(splits.X_test):,} samples")
    print(f"Features: {list(splits.X_train.columns)}")
