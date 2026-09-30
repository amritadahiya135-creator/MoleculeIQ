"""
MoleculeIQ - Step 5: Explainability (SHAP)
------------------------------------------------------------------------
Explains WHY the toxicity model made a given prediction, rather than
just outputting a score. Uses SHAP to identify which fingerprint bits
(i.e. which molecular substructures) pushed the prediction toward
"toxic", then visualizes those substructures directly on the molecule.

Run:  python 05_explainability.py
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import joblib
import shap


def get_fingerprint_with_bitinfo(smiles: str, n_bits: int = 2048, radius: int = 2):
    """
    Compute a Morgan fingerprint AND keep track of which substructure
    each 'on' bit corresponds to (bitInfo) -- this is what lets us
    later draw the actual chemical structure behind a SHAP value,
    instead of just showing a meaningless bit number.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )
    arr = np.array(fp).reshape(1, -1)
    return mol, arr, bit_info


def explain_prediction(smiles: str, model, top_n: int = 5):
    """
    Runs SHAP on a single molecule's prediction and reports the top
    contributing fingerprint bits (positive = pushes toward "toxic").
    """
    mol, fp_array, bit_info = get_fingerprint_with_bitinfo(smiles)
    if mol is None:
        print(f"  [!] Could not parse SMILES: '{smiles}'")
        return

    prob_toxic = model.predict_proba(fp_array)[0][1]
    print(f"\nPredicted toxicity probability: {prob_toxic:.3f}")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(fp_array, check_additivity=False)

    # explanation.values shape is (1, n_bits, n_classes) for this model type
    # -- we want the contributions toward class 1 ("toxic")
    shap_values = explanation.values[0, :, 1]

    # Only bits that are actually "on" (present) in this molecule matter
    on_bits = [b for b in range(len(shap_values)) if fp_array[0, b] == 1]
    on_bits_sorted = sorted(on_bits, key=lambda b: abs(shap_values[b]), reverse=True)
    top_bits = on_bits_sorted[:top_n]

    print(f"\nTop {top_n} structural features driving this prediction:")
    print(f"{'Bit':>6}  {'SHAP value':>11}  {'Effect'}")
    for b in top_bits:
        val = shap_values[b]
        direction = "-> pushes toward TOXIC" if val > 0 else "-> pushes toward NON-TOXIC"
        print(f"  {b:>4}  {val:>11.4f}  {direction}")

    # Visualize the top substructures directly on the molecule
    try:
        highlight_bits = [(b, bit_info[b]) for b in top_bits if b in bit_info]
        img = Draw.DrawMorganBits(
            [(mol, b, bit_info) for b, _ in highlight_bits],
            molsPerRow=min(len(highlight_bits), 5),
            legends=[f"bit {b} ({shap_values[b]:+.3f})" for b, _ in highlight_bits],
        )
        if isinstance(img, str):
            # RDKit returned SVG markup instead of a PNG image object
            out_path = "explanation_substructures.svg"
            with open(out_path, "w") as f:
                f.write(img)
        else:
            out_path = "explanation_substructures.png"
            img.save(out_path)
        print(f"\nSubstructure visualization saved to {out_path}")
    except Exception as e:
        print(f"\n  [note] Could not render substructure image ({e}) -- "
              f"numeric explanation above is still valid.")


if __name__ == "__main__":
    print("Loading toxicity model...")
    model = joblib.load("toxicity_model.joblib")

    test_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    }

    for name, smiles in test_molecules.items():
        print("\n" + "=" * 60)
        print(f"Explaining prediction for: {name}")
        print("=" * 60)
        explain_prediction(smiles, model)