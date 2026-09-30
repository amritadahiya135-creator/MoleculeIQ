"""
MoleculeIQ - Step 1: SMILES Parsing & Molecular Descriptor Calculation
------------------------------------------------------------------------
This is the foundation of the whole pipeline. Every later component
(toxicity model, similarity search, ranking) builds on being able to
reliably turn a SMILES string into a structured set of features.

Run:  python 01_descriptors.py
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen


def parse_molecule(smiles: str):
    """
    Convert a SMILES string into an RDKit molecule object.
    Returns None if the SMILES is invalid -- ALWAYS check for this,
    real-world input (user-typed or batch CSVs) will contain garbage.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [!] Could not parse SMILES: '{smiles}' -- skipping.")
        return None
    return mol


def compute_descriptors(mol) -> dict:
    """
    Compute the core set of descriptors we'll show in the dashboard.
    These map directly to the "Molecular Analysis" panel from the
    project plan (molecular weight, LogP, solubility proxy, drug-likeness).
    """
    mw = Descriptors.MolWt(mol)                     # Molecular weight
    logp = Crippen.MolLogP(mol)                      # Lipophilicity (LogP)
    h_donors = Lipinski.NumHDonors(mol)               # H-bond donors
    h_acceptors = Lipinski.NumHAcceptors(mol)         # H-bond acceptors
    rot_bonds = Lipinski.NumRotatableBonds(mol)       # Flexibility
    tpsa = Descriptors.TPSA(mol)                      # Topological polar surface area
                                                       # (used as a rough solubility/
                                                       # permeability proxy)

    # Lipinski's Rule of Five: a classic, well-established heuristic for
    # "drug-likeness". A molecule violating >1 rule is less likely to be
    # orally bioavailable. This is NOT a toxicity or efficacy prediction --
    # it's a simple structural filter, and we present it as exactly that.
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if h_donors > 5:
        violations += 1
    if h_acceptors > 10:
        violations += 1

    return {
        "molecular_weight": round(mw, 2),
        "logP": round(logp, 2),
        "h_bond_donors": h_donors,
        "h_bond_acceptors": h_acceptors,
        "rotatable_bonds": rot_bonds,
        "tpsa": round(tpsa, 2),
        "lipinski_violations": violations,
        "drug_like": violations <= 1,   # standard Lipinski threshold
    }


def analyze_smiles(smiles: str):
    """End-to-end: parse + compute descriptors for one molecule."""
    mol = parse_molecule(smiles)
    if mol is None:
        return None
    descriptors = compute_descriptors(mol)
    descriptors["smiles"] = smiles
    return descriptors


if __name__ == "__main__":
    # A few real, well-known molecules to sanity-check the pipeline against
    test_molecules = {
        "Ethanol": "CCO",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Invalid example": "not_a_real_smiles",
    }

    print("=" * 60)
    print("MoleculeIQ -- Descriptor Calculation Test")
    print("=" * 60)

    for name, smiles in test_molecules.items():
        print(f"\n{name}  (SMILES: {smiles})")
        result = analyze_smiles(smiles)
        if result is None:
            continue
        for key, value in result.items():
            if key == "smiles":
                continue
            print(f"    {key:22s}: {value}")