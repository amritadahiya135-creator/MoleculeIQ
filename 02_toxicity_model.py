"""
MoleculeIQ - Step 2: Toxicity Classifier (ClinTox)
------------------------------------------------------------------------
Trains a binary classifier to predict clinical-trial toxicity failure
(CT_TOX) from a molecule's structure, using Morgan fingerprints as
features. This is the core ML component of the project.

Run:  python 02_toxicity_model.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib


def load_clintox() -> pd.DataFrame:
    """
    Load the ClinTox dataset directly from its public CSV -- no deepchem
    or TensorFlow required. Contains SMILES strings plus two label
    columns; we use CT_TOX (failed clinical trials due to toxicity).
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    df = pd.read_csv(url)
    df = df[["smiles", "CT_TOX"]].dropna()
    df["CT_TOX"] = df["CT_TOX"].astype(int)
    return df


def smiles_to_fingerprint(smiles: str, n_bits: int = 2048):
    """
    Convert a SMILES string into a Morgan fingerprint (ECFP4-equivalent) --
    the standard molecular representation for ML in cheminformatics.
    Returns None for unparseable SMILES so the caller can filter it out.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)


def build_feature_matrix(df: pd.DataFrame):
    """Turn every molecule in the dataframe into a fingerprint feature row."""
    features, labels, kept_smiles = [], [], []
    dropped = 0

    for smiles, label in zip(df["smiles"], df["CT_TOX"]):
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            dropped += 1
            continue
        features.append(fp)
        labels.append(label)
        kept_smiles.append(smiles)

    print(f"  Parsed {len(features)} molecules successfully, "
          f"dropped {dropped} invalid SMILES.")
    return np.array(features), np.array(labels), kept_smiles


def train_and_evaluate(X, y):
    """
    Train a Random Forest classifier and report HONEST performance on a
    held-out test set. These are the numbers to put in your report/slides --
    never substitute training accuracy for this.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET PERFORMANCE (report these numbers)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["Non-toxic", "Toxic"]))

    return clf, (X_test, y_test)


if __name__ == "__main__":
    print("Loading ClinTox dataset...")
    df = load_clintox()
    print(f"  Loaded {len(df)} labeled molecules.")
    print(f"  Class balance: {df['CT_TOX'].value_counts().to_dict()}")

    print("\nComputing Morgan fingerprints...")
    X, y, kept_smiles = build_feature_matrix(df)

    print("\nTraining Random Forest classifier...")
    model, test_data = train_and_evaluate(X, y)

    joblib.dump(model, "toxicity_model.joblib")
    print("\nModel saved to toxicity_model.joblib")