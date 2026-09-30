"""
MoleculeIQ - Step 6 (Phase 2): Multi-Label Tox21 Toxicity Model
------------------------------------------------------------------------
Extends toxicity prediction from a single ClinTox endpoint to 12 real
Tox21 assay endpoints (nuclear receptor and stress-response pathways),
giving a much richer toxicity profile per molecule than a single
toxic/non-toxic label.

Tracks every training run with MLflow so metrics, parameters, and the
model itself are logged automatically -- real experiment tracking, not
just a single ad-hoc training script.

Run:  python 06_tox21_model.py
Then: mlflow ui   (in a separate terminal, to view results in browser)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import mlflow
import mlflow.sklearn
import joblib


TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

# The 12 real Tox21 assay endpoints -- each is a separate toxicity-
# relevant biological pathway, not just "toxic/not toxic"
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def load_tox21() -> pd.DataFrame:
    """Load the Tox21 dataset directly from its public CSV."""
    df = pd.read_csv(TOX21_URL)
    return df


def smiles_to_fp_array(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)


def build_features(df: pd.DataFrame):
    """
    Build the feature matrix and a label matrix (one column per Tox21
    task). Tox21 has real missing values per task (not every compound
    was tested against every assay) -- we keep NaN as-is per task and
    handle it during per-task training, rather than dropping rows that
    are only partially labeled.
    """
    features, kept_rows = [], []
    dropped = 0

    for idx, row in df.iterrows():
        fp = smiles_to_fp_array(row["smiles"])
        if fp is None:
            dropped += 1
            continue
        features.append(fp)
        kept_rows.append(idx)

    print(f"  Parsed {len(features)} molecules successfully, dropped {dropped} invalid SMILES.")
    X = np.array(features)
    labels_df = df.loc[kept_rows, TOX21_TASKS].reset_index(drop=True)
    return X, labels_df


def train_task_model(X, y_task, task_name: str):
    """
    Train and evaluate one Random Forest per Tox21 task. Rows with a
    missing label for this specific task are excluded from that task's
    training only (standard practice for Tox21's partially-labeled data).
    """
    mask = ~y_task.isna()
    X_task = X[mask]
    y_task = y_task[mask].astype(int)

    if y_task.nunique() < 2 or y_task.sum() < 5:
        print(f"  [!] Skipping {task_name} -- insufficient positive examples.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
    )

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": float(y_task.mean()),
    }
    return clf, metrics


if __name__ == "__main__":
    mlflow.set_experiment("moleculeiq-tox21")

    print("Loading Tox21 dataset...")
    df = load_tox21()
    print(f"  Loaded {len(df)} compounds with up to {len(TOX21_TASKS)} assay labels each.")

    print("\nComputing Morgan fingerprints (this covers all tasks, done once)...")
    X, labels_df = build_features(df)

    models = {}
    summary_rows = []

    for task in TOX21_TASKS:
        print(f"\n{'=' * 60}\nTraining task: {task}\n{'=' * 60}")

        with mlflow.start_run(run_name=task):
            result = train_task_model(X, labels_df[task], task)
            if result is None:
                mlflow.log_param("status", "skipped_insufficient_data")
                continue

            clf, metrics = result
            models[task] = clf

            mlflow.log_param("task", task)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 300)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(
                clf, name=f"model_{task}",
                serialization_format="cloudpickle",
            )

            print(f"  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  "
                  f"F1: {metrics['f1']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}")

            summary_rows.append({"task": task, **metrics})

    joblib.dump(models, "tox21_multilabel_models.joblib")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("tox21_results_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL TOX21 TASKS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nAll {len(models)} task models saved to tox21_multilabel_models.joblib")
    print("Full experiment log available via: mlflow ui")