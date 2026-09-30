"""
MoleculeIQ - Step 4: Batch Ranking Engine
------------------------------------------------------------------------
Ties everything together: given a CSV of candidate molecules, computes
descriptors, predicts toxicity risk, finds similarity to known compounds,
and produces a single ranked output table.

This is the centerpiece of the project -- the actual "tool" a researcher
would use.

Run:  python 04_ranking_engine.py sample_candidates.csv
"""

import sys
import importlib.util
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib


def load_module(filename, module_name):
    """
    Python module names can't start with a digit, so our numbered
    filenames (01_descriptors.py etc.) can't be imported the normal way.
    This loads them directly from their file path instead.
    """
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


descriptors_mod = load_module("01_descriptors.py", "descriptors_mod")
similarity_mod = load_module("03_similarity_search.py", "similarity_mod")


def smiles_to_fp_array(smiles: str, n_bits: int = 2048):
    """Same fingerprint format the toxicity model was trained on."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp).reshape(1, -1)


def predict_toxicity(smiles: str, model):
    """Returns predicted probability of toxicity (0-1), or None if invalid."""
    fp = smiles_to_fp_array(smiles)
    if fp is None:
        return None
    return model.predict_proba(fp)[0][1]  # probability of class "1" (toxic)


def rank_candidates(csv_path: str, output_path: str = "ranked_candidates.csv"):
    print(f"Loading candidates from {csv_path}...")
    candidates = pd.read_csv(csv_path)

    print("Loading toxicity model...")
    tox_model = joblib.load("toxicity_model.joblib")

    print("Loading similarity reference library (uses cache if available)...")
    reference = similarity_mod.build_reference_fingerprints()

    rows = []
    for _, row in candidates.iterrows():
        name = row.get("name", row["smiles"])
        smiles = row["smiles"]

        mol = descriptors_mod.parse_molecule(smiles)
        if mol is None:
            print(f"  [!] Skipping '{name}' -- invalid SMILES.")
            continue

        desc = descriptors_mod.compute_descriptors(mol)
        tox_prob = predict_toxicity(smiles, tox_model)

        sim_results = similarity_mod.find_similar(smiles, top_n=1, reference=reference)
        best_similarity = sim_results[0][1] if sim_results else 0.0

        # --- Overall score (0-100, higher = more promising to investigate) ---
        # This is a simple, explainable weighted combination, not a
        # trained model -- intentionally transparent for a first version:
        #   - Lower predicted toxicity risk is good      (50% weight)
        #   - Passing Lipinski drug-likeness is good      (30% weight)
        #   - Similarity to known compounds is a mild plus (20% weight),
        #     since being close to known, studied chemical space is a
        #     weak positive signal, not a guarantee of activity.
        safety_score = (1 - tox_prob) * 100
        druglike_score = 100 if desc["drug_like"] else 40
        similarity_score = best_similarity * 100

        overall = (safety_score * 0.5) + (druglike_score * 0.3) + (similarity_score * 0.2)

        rows.append({
            "name": name,
            "smiles": smiles,
            "molecular_weight": desc["molecular_weight"],
            "logP": desc["logP"],
            "drug_like": desc["drug_like"],
            "predicted_toxicity_risk": round(tox_prob, 3),
            "best_known_similarity": round(best_similarity, 3),
            "overall_score": round(overall, 1),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))

    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("RANKED CANDIDATES")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print(f"\nSaved to {output_path}")

    return result_df


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_candidates.csv"
    rank_candidates(csv_path)