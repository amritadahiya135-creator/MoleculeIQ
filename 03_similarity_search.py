"""
MoleculeIQ - Step 3: Molecular Similarity Search
------------------------------------------------------------------------
Given a query molecule, finds the most structurally similar compounds
in a reference library using Morgan fingerprints + Tanimoto similarity.

Reference library: approved small-molecule drugs pulled from ChEMBL --
a real, well-defined set of known drugs (not the full ~2.4M-compound
ChEMBL database, which is impractical to fingerprint and search locally).

Run:  python 03_similarity_search.py
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import joblib


REFERENCE_CACHE = "chembl_reference_library.joblib"
APPROVED_DRUGS_CSV = "chembl_approved_drugs.csv"


def load_reference_library() -> pd.DataFrame:
    """
    Load approved drugs from ChEMBL to use as the similarity search
    library -- a real, well-defined reference set of known drugs,
    rather than the full ~2.4M-compound ChEMBL database (impractical
    to fingerprint and search locally).
    """
    from chembl_webresource_client.new_client import new_client

    print("  Querying ChEMBL for approved small-molecule drugs "
          "(first run only -- this can take a few minutes)...")
    molecule = new_client.molecule
    approved = molecule.filter(
        max_phase=4, molecule_type="Small molecule"
    ).only(["molecule_chembl_id", "molecule_structures"])

    smiles_list = []
    for entry in approved:
        structures = entry.get("molecule_structures")
        if structures and structures.get("canonical_smiles"):
            smiles_list.append(structures["canonical_smiles"])

    df = pd.DataFrame({"smiles": smiles_list}).dropna().drop_duplicates()
    print(f"  Retrieved {len(df)} approved drug structures from ChEMBL.")

    # Cache the raw compound list too, separate from the fingerprint
    # cache, so re-running doesn't hit the ChEMBL API every time.
    df.to_csv(APPROVED_DRUGS_CSV, index=False)
    return df


def load_reference_library_cached() -> pd.DataFrame:
    """Use the cached CSV if we already pulled ChEMBL once; otherwise fetch it."""
    if os.path.exists(APPROVED_DRUGS_CSV):
        return pd.read_csv(APPROVED_DRUGS_CSV)
    return load_reference_library()


def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute a Morgan fingerprint object (not a plain array -- RDKit's
    bulk similarity functions expect the native fingerprint type)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def build_reference_fingerprints(force_rebuild: bool = False):
    """
    Build (or load from cache) fingerprints for every molecule in the
    reference library. This is the slow step, so we cache it -- no need
    to recompute thousands of fingerprints every single run.
    """
    if os.path.exists(REFERENCE_CACHE) and not force_rebuild:
        print(f"Loading cached reference library from {REFERENCE_CACHE}...")
        return joblib.load(REFERENCE_CACHE)

    print("Building reference library fingerprints (first run, may take a bit)...")
    df = load_reference_library_cached()

    smiles_list, fps = [], []
    dropped = 0
    for smiles in df["smiles"]:
        fp = compute_fingerprint(smiles)
        if fp is None:
            dropped += 1
            continue
        smiles_list.append(smiles)
        fps.append(fp)

    print(f"  Built fingerprints for {len(fps)} compounds, dropped {dropped} invalid SMILES.")

    data = {"smiles": smiles_list, "fingerprints": fps}
    joblib.dump(data, REFERENCE_CACHE)
    return data


def find_similar(query_smiles: str, top_n: int = 5, reference=None):
    """
    Given a query molecule, return the top_n most similar compounds in
    the reference library, ranked by Tanimoto similarity (0-1, higher =
    more similar).
    """
    if reference is None:
        reference = build_reference_fingerprints()

    query_fp = compute_fingerprint(query_smiles)
    if query_fp is None:
        print(f"  [!] Could not parse query SMILES: '{query_smiles}'")
        return []

    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, reference["fingerprints"])

    results = list(zip(reference["smiles"], similarities))
    results.sort(key=lambda x: x[1], reverse=True)

    # Skip the query molecule itself if it happens to already be in the
    # reference set (similarity would be a trivial 1.0)
    results = [r for r in results if r[1] < 0.999]

    return results[:top_n]


if __name__ == "__main__":
    reference = build_reference_fingerprints()

    test_queries = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    }

    for name, smiles in test_queries.items():
        print(f"\n{'=' * 60}")
        print(f"Query: {name}  ({smiles})")
        print("=" * 60)
        results = find_similar(smiles, top_n=5, reference=reference)
        for i, (ref_smiles, score) in enumerate(results, 1):
            print(f"  {i}. Similarity {score:.3f}  --  {ref_smiles}")