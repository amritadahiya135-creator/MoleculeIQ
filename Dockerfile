# MoleculeIQ - Dockerfile
# ---------------------------------------------------------------------
# Packages the full app (descriptors, toxicity model, similarity search,
# ranking, explainability, Streamlit UI) into a single reproducible
# container.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer -- speeds up rebuilds when
# only your code changes, not your requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and pre-built artifacts (trained model,
# cached reference library) so the container works immediately without
# needing to retrain or re-query ChEMBL on first launch
COPY 01_descriptors.py .
COPY 02_toxicity_model.py .
COPY 03_similarity_search.py .
COPY 04_ranking_engine.py .
COPY 05_explainability.py .
COPY app.py .
COPY sample_candidates.csv .
COPY toxicity_model.joblib .
COPY chembl_reference_library.joblib .
COPY chembl_approved_drugs.csv .

EXPOSE 8501

# --server.address=0.0.0.0 is required -- without it, Streamlit only
# listens inside the container and you can't reach it from your browser
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]