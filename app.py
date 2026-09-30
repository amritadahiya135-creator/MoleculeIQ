"""
MoleculeIQ - Streamlit App
------------------------------------------------------------------------
The user-facing interface tying together descriptors, toxicity
prediction, similarity search, ranking, and explainability into one
interactive tool.

Run:  streamlit run app.py
"""

import importlib.util
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def load_module(filename, module_name):
    """Load our numbered scripts as modules (filenames can't start with
    a digit for a normal import, so we load them by file path instead)."""
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def load_everything():
    """Load the model, reference library, and helper modules once and
    cache them -- avoids re-loading on every single user interaction."""
    descriptors_mod = load_module("01_descriptors.py", "descriptors_mod")
    similarity_mod = load_module("03_similarity_search.py", "similarity_mod")
    tox_model = joblib.load("toxicity_model.joblib")
    reference = similarity_mod.build_reference_fingerprints()
    return descriptors_mod, similarity_mod, tox_model, reference


def smiles_to_fp_array(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp).reshape(1, -1)


def get_shap_explanation(smiles: str, model, top_n: int = 8):
    """Returns a small DataFrame of the top contributing fingerprint
    bits for a single molecule's toxicity prediction."""
    fp_array = smiles_to_fp_array(smiles)
    if fp_array is None:
        return None
    explainer = shap.TreeExplainer(model)
    explanation = explainer(fp_array, check_additivity=False)
    shap_values = explanation.values[0, :, 1]

    on_bits = [b for b in range(len(shap_values)) if fp_array[0, b] == 1]
    on_bits_sorted = sorted(on_bits, key=lambda b: abs(shap_values[b]), reverse=True)
    top_bits = on_bits_sorted[:top_n]

    return pd.DataFrame({
        "fingerprint_bit": [str(b) for b in top_bits],
        "shap_value": [shap_values[b] for b in top_bits],
    }).set_index("fingerprint_bit")


def analyze_single_molecule(smiles, descriptors_mod, similarity_mod, tox_model):
    mol = descriptors_mod.parse_molecule(smiles)
    if mol is None:
        return None

    desc = descriptors_mod.compute_descriptors(mol)
    fp_array = smiles_to_fp_array(smiles)
    tox_prob = tox_model.predict_proba(fp_array)[0][1]
    similar = similarity_mod.find_similar(smiles, top_n=5)

    return desc, tox_prob, similar


def rank_batch(df, descriptors_mod, similarity_mod, tox_model, reference):
    rows = []
    for _, row in df.iterrows():
        name = row.get("name", row["smiles"])
        smiles = row["smiles"]
        mol = descriptors_mod.parse_molecule(smiles)
        if mol is None:
            continue

        desc = descriptors_mod.compute_descriptors(mol)
        fp_array = smiles_to_fp_array(smiles)
        tox_prob = tox_model.predict_proba(fp_array)[0][1]
        sim_results = similarity_mod.find_similar(smiles, top_n=1, reference=reference)
        best_similarity = sim_results[0][1] if sim_results else 0.0

        safety_score = (1 - tox_prob) * 100
        druglike_score = 100 if desc["drug_like"] else 40
        similarity_score = best_similarity * 100
        overall = (safety_score * 0.5) + (druglike_score * 0.3) + (similarity_score * 0.2)

        rows.append({
            "name": name, "smiles": smiles,
            "molecular_weight": desc["molecular_weight"], "logP": desc["logP"],
            "drug_like": desc["drug_like"],
            "predicted_toxicity_risk": round(tox_prob, 3),
            "best_known_similarity": round(best_similarity, 3),
            "overall_score": round(overall, 1),
        })

    result_df = pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="MoleculeIQ", layout="wide")
st.title("MoleculeIQ")
st.caption("A computational screening and prioritization assistant — not a drug-discovery system. "
           "All predictions are computational estimates requiring experimental validation.")

descriptors_mod, similarity_mod, tox_model, reference = load_everything()

tab1, tab2 = st.tabs(["Single Molecule", "Batch Screening"])

with tab1:
    st.subheader("Analyze a single molecule")
    smiles_input = st.text_input("SMILES string", value="CC(=O)OC1=CC=CC=C1C(=O)O",
                                  help="Example shown is Aspirin")

    if st.button("Analyze", key="single_analyze"):
        result = analyze_single_molecule(smiles_input, descriptors_mod, similarity_mod, tox_model)
        if result is None:
            st.error("Could not parse this SMILES string. Please check it and try again.")
        else:
            desc, tox_prob, similar = result

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Toxicity Risk", f"{tox_prob:.1%}")
                st.metric("Drug-likeness (Lipinski)", "Pass" if desc["drug_like"] else "Fail")
            with col2:
                st.metric("Molecular Weight", f"{desc['molecular_weight']}")
                st.metric("LogP", f"{desc['logP']}")

            st.markdown("**Full descriptors**")
            st.json(desc)

            st.markdown("**Most similar known compounds**")
            sim_df = pd.DataFrame(similar, columns=["smiles", "similarity"])
            st.dataframe(sim_df, width="stretch")

            st.markdown("**Why this toxicity prediction? (top contributing molecular features)**")
            shap_df = get_shap_explanation(smiles_input, tox_model)
            if shap_df is not None:
                st.bar_chart(shap_df["shap_value"])
                st.caption("Positive values push the prediction toward 'toxic'; "
                           "negative values push toward 'non-toxic'.")

with tab2:
    st.subheader("Batch screening")
    st.write("Upload a CSV with columns `name` (optional) and `smiles`.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        with st.spinner("Running descriptors, toxicity prediction, and similarity search..."):
            result_df = rank_batch(input_df, descriptors_mod, similarity_mod, tox_model, reference)

        st.success(f"Ranked {len(result_df)} candidates.")
        st.dataframe(result_df, width="stretch")

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download ranked results as CSV", csv_out,
                            file_name="ranked_candidates.csv", mime="text/csv")