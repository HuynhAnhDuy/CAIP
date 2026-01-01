import sys
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from fingerprints import (
    canonicalize_smiles,
    get_ecfp,
    get_maccs,
    get_rdkit,
    safe_fp,
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = [
    # XGB models
    {
        "name": "XGB-ECFP",
        "model_type": "xgb",
        "fingerprint": "ecfp",
        "path": "models/xgb_model_ecfp.pkl",
    },
    {
        "name": "XGB-MACCS",
        "model_type": "xgb",
        "fingerprint": "maccs",
        "path": "models/xgb_model_maccs.pkl",
    },
    {
        "name": "XGB-RDKit",
        "model_type": "xgb",
        "fingerprint": "rdkit",
        "path": "models/xgb_model_rdkit.pkl",
    },
    # BiLSTM models
    {
        "name": "BiLSTM-ECFP",
        "model_type": "bilstm",
        "fingerprint": "ecfp",
        "path": "models/bilstm_ecfp.keras",
    },
    {
        "name": "BiLSTM-MACCS",
        "model_type": "bilstm",
        "fingerprint": "maccs",
        "path": "models/bilstm_maccs.keras",
    },
    {
        "name": "BiLSTM-RDKit",
        "model_type": "bilstm",
        "fingerprint": "rdkit",
        "path": "models/bilstm_rdkit.keras",
    },
]


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================

@st.cache_resource
def load_models():
    """
    Load all XGB and BiLSTM models into a dictionary.
    Key: model name (e.g., 'XGB-ECFP'); Value: model object.
    """
    models = {}
    for cfg in MODEL_CONFIG:
        name = cfg["name"]
        path = cfg["path"]
        if cfg["model_type"] == "xgb":
            with open(path, "rb") as f:
                model = pickle.load(f)
        else:  # BiLSTM
            model = load_model(path)
        models[name] = model
    return models


# ============================================================
# FEATURE GENERATION FROM SMILES
# ============================================================

def featurize_smiles(smiles: str):
    """
    From a raw SMILES string:
        - Convert to canonical SMILES
        - Compute ECFP, MACCS, and RDKit fingerprints
    """
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None, "Invalid SMILES: canonicalization failed."

    ecfp = safe_fp(get_ecfp(canonical))
    maccs = safe_fp(get_maccs(canonical))
    rdkit_fp = safe_fp(get_rdkit(canonical))

    if ecfp is None or maccs is None or rdkit_fp is None:
        return None, "Failed to generate full fingerprints (ECFP/MACCS/RDKit) from canonical SMILES."

    features = {
        "canonical": canonical,
        "ecfp": ecfp,
        "maccs": maccs,
        "rdkit": rdkit_fp,
    }
    return features, None


# ============================================================
# PREDICTION HELPERS
# ============================================================

def predict_with_xgb(model, fp_vec: np.ndarray) -> float:
    """
    Predict probability (class = 1) using an XGBClassifier.
    """
    x = fp_vec.reshape(1, -1)
    proba = model.predict_proba(x)[0, 1]
    return float(proba)


def predict_with_bilstm(model, fp_vec: np.ndarray) -> float:
    """
    Predict probability using a BiLSTM model.

    Training shape:
        x_train -> (n_samples, 1, n_features)
        input_shape=(1, input_dim)

    Inference shape:
        (1, 1, n_features)
    """
    x = fp_vec.reshape(1, 1, -1)
    proba = model.predict(x, verbose=0).ravel()[0]
    return float(proba)


def label_from_probability(prob: float, threshold: float = 0.5):
    """
    Map probability to a binary label.
    """
    if prob > threshold:
        return True, "Anti-inflammatory activity"
    else:
        return False, "Non-anti-inflammatory activity"


# ============================================================
# CONSENSUS LOGIC
# ============================================================

def consensus_from_results(results):
    """
    Compute a consensus outcome from multiple models.

    Rules:
    - All 6 active      -> "Anti-inflammatory activity"
    - All 6 non-active  -> "Non-anti-inflammatory activity"
    - Otherwise         -> "Inconclusive"
    """
    active_flags = [r["is_active"] for r in results]
    if all(active_flags):
        return "Anti-inflammatory activity"
    elif not any(active_flags):
        return "Non-anti-inflammatory activity"
    else:
        return "Inconclusive"


# ============================================================
# HELPER: RUN ALL MODELS FOR ONE MOLECULE
# ============================================================

def run_all_models_on_features(models, feature_dict):
    """
    Run all 6 models on a single compound.

    feature_dict keys:
        - 'ecfp', 'maccs', 'rdkit'
    Returns
    -------
    results : list[dict]
        Each dict has keys:
            'Model', 'Type', 'Fingerprint',
            'Probability', 'Decision', 'is_active'
    """
    ecfp_fp = feature_dict["ecfp"]
    maccs_fp = feature_dict["maccs"]
    rdkit_fp = feature_dict["rdkit"]

    results = []
    for cfg in MODEL_CONFIG:
        name = cfg["name"]
        model_type = cfg["model_type"]
        fp_name = cfg["fingerprint"]
        model = models[name]

        if fp_name == "ecfp":
            vec = ecfp_fp
        elif fp_name == "maccs":
            vec = maccs_fp
        elif fp_name == "rdkit":
            vec = rdkit_fp
        else:
            continue

        if model_type == "xgb":
            prob = predict_with_xgb(model, vec)
        else:  # BiLSTM
            prob = predict_with_bilstm(model, vec)

        is_active, label = label_from_probability(prob, threshold=0.5)

        results.append(
            {
                "Model": name,
                "Type": model_type.upper(),
                "Fingerprint": fp_name.upper(),
                "Probability": prob,
                "Decision": label,
                "is_active": is_active,
            }
        )

    return results


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="CAIP – Consensus Anti-inflammatory Predictor",
        layout="centered",
    )

    # === Sidebar Instructions ===
    with st.sidebar:
        st.title("CAIP web tool")
        st.markdown(
            "### How to use\n"
            "**Single prediction**\n"
            "1. Enter a valid SMILES string in the input box.\n"
            "2. Click **Predict** to run all 6 models.\n"
            "3. Review the per-model probabilities and activity labels.\n"
            "4. Check the **consensus outcome** at the bottom.\n\n"
            "**Batch prediction**\n"
            "1. Upload a CSV file containing one column of SMILES.\n"
            "2. Select the SMILES column.\n"
            "3. Click **Run batch prediction**.\n"
            "4. Download the prediction results as CSV.\n\n"
            "**Decision rule (per model):**\n"
            "- If probability ≥ 0.5 → classified as **Anti-inflammatory activity**.\n"
            "- If probability < 0.5 → classified as **Non-anti-inflammatory activity**.\n\n"
            "**CAIP consensus decision:**\n"
            "- If all 6 models predict *Anti-inflammatory activity* → final label: **Anti-inflammatory activity**.\n"
            "- If all 6 models predict *Non-anti-inflammatory activity* → final label: **Non-anti-inflammatory activity**.\n"
            "- If there is any disagreement between models → final label: **Inconclusive**."
        )

    st.title("CAIP: Consensus Anti-inflammatory Predictor")

    st.markdown(
        "**This web application leverages a consensus of six machine learning and deep learning models "
        "to predict the anti-inflammatory potential of compounds from their SMILES representations.**\n\n"
        "**Workflow:**\n"
        "1. Convert the user-input SMILES to canonical SMILES.\n"
        "2. Compute ECFP, MACCS, and RDKit fingerprints.\n"
        "3. Run six predictive models: three XGBoost classifiers and three BiLSTM neural networks.\n"
        "4. Display individual model probabilities and per-model labels.\n"
        "5. Compute a consensus outcome across all six models to determine the final prediction for each compound. \n\n"
    )
    
    st.markdown(
    "<p style='color:red;'>"
    "<b>Important note:</b> By default, CAIP adopts a strict unanimity criterion "
    "(all six models must agree). Users may optionally relax this requirement "
    "to a majority-vote rule (e.g., at least 3 out of 6 models predicting activity)."
    "</p>",
    unsafe_allow_html=True,
    )

    st.markdown("---")

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

    # ========================================================
    # SINGLE PREDICTION TAB
    # ========================================================
    with tab_single:
        smiles_input = st.text_input(
            "SMILES input",
            value="",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
        )

        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("Predict", type="primary")
        with col2:
            example_button = st.button("Use example")

        if example_button:
            smiles_input = "CC(=O)Oc1ccccc1C(=O)O"
            st.session_state["smiles_example"] = smiles_input

        # Re-display example if stored
        if "smiles_example" in st.session_state and st.session_state["smiles_example"]:
            smiles_input = st.text_input(
                "SMILES input (example loaded)",
                value=st.session_state["smiles_example"],
                placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
            )

        if run_button:
            if not smiles_input.strip():
                st.warning("Please enter a SMILES string.")
            else:
                with st.spinner("Loading models..."):
                    models = load_models()

                with st.spinner("Canonicalizing SMILES and computing fingerprints..."):
                    features, err = featurize_smiles(smiles_input)

                if err is not None:
                    st.error(err)
                else:
                    canonical = features["canonical"]
                    st.success("Fingerprint generation successful.")
                    st.markdown(f"**Canonical SMILES:** `{canonical}`")
                    st.markdown("---")

                    # Run all 6 models
                    results = run_all_models_on_features(models, features)

                    # Display per-model results
                    df_show = pd.DataFrame(
                        [
                            {
                                "Model": r["Model"],
                                "Type": r["Type"],
                                "Fingerprint": r["Fingerprint"],
                                "Probability": f"{r['Probability']:.4f}",
                                "Decision": r["Decision"],
                            }
                            for r in results
                        ]
                    )

                    st.subheader("Individual model predictions")
                    st.table(df_show)

                    # Consensus
                    consensus = consensus_from_results(results)

                    st.markdown("---")
                    st.subheader("Consensus outcome (all 6 models)")
                    if consensus == "Inconclusive":
                        st.warning("**Outcome: Inconclusive** (disagreement among models).")
                    elif consensus == "Anti-inflammatory activity":
                        st.success(
                            "**Outcome: Anti-inflammatory activity** "
                            "(all 6 models predict activity)."
                        )
                    else:
                        st.info(
                            "**Outcome: Non-anti-inflammatory activity** "
                            "(all 6 models predict no activity)."
                        )

                    st.caption(
                        "Classification rule: probability > 0.5 → Anti-inflammatory activity; "
                        "otherwise → Non-anti-inflammatory activity."
                    )

    # ========================================================
    # BATCH PREDICTION TAB
    # ========================================================
    with tab_batch:
        st.subheader("Batch prediction")

        uploaded_file = st.file_uploader(
            "Upload a CSV file containing a SMILES column",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")
                df_input = None

            if df_input is not None:
                st.write("Preview of uploaded data:")
                st.dataframe(df_input.head())

                if df_input.shape[1] == 0:
                    st.error("The uploaded CSV has no columns.")
                else:
                    smi_col = st.selectbox(
                        "Select the SMILES column",
                        options=list(df_input.columns),
                    )

                    run_batch = st.button("Run batch prediction")

                    if run_batch:
                        with st.spinner("Loading models..."):
                            models = load_models()

                        results_rows = []

                        for idx, row in df_input.iterrows():
                            raw_smiles = str(row[smi_col]) if pd.notna(row[smi_col]) else ""
                            if not raw_smiles.strip():
                                results_rows.append(
                                    {
                                        "Input_SMILES": raw_smiles,
                                        "Canonical_SMILES": None,
                                        "XGB_ECFP_prob": None,
                                        "XGB_ECFP_label": None,
                                        "XGB_MACCS_prob": None,
                                        "XGB_MACCS_label": None,
                                        "XGB_RDKIT_prob": None,
                                        "XGB_RDKIT_label": None,
                                        "BiLSTM_ECFP_prob": None,
                                        "BiLSTM_ECFP_label": None,
                                        "BiLSTM_MACCS_prob": None,
                                        "BiLSTM_MACCS_label": None,
                                        "BiLSTM_RDKIT_prob": None,
                                        "BiLSTM_RDKIT_label": None,
                                        "Consensus_Label": "Invalid_SMILES",
                                        "Error": "Empty or missing SMILES",
                                    }
                                )
                                continue

                            features, err = featurize_smiles(raw_smiles)
                            if err is not None:
                                # Invalid SMILES or fingerprint error
                                results_rows.append(
                                    {
                                        "Input_SMILES": raw_smiles,
                                        "Canonical_SMILES": None,
                                        "XGB_ECFP_prob": None,
                                        "XGB_ECFP_label": None,
                                        "XGB_MACCS_prob": None,
                                        "XGB_MACCS_label": None,
                                        "XGB_RDKIT_prob": None,
                                        "XGB_RDKIT_label": None,
                                        "BiLSTM_ECFP_prob": None,
                                        "BiLSTM_ECFP_label": None,
                                        "BiLSTM_MACCS_prob": None,
                                        "BiLSTM_MACCS_label": None,
                                        "BiLSTM_RDKIT_prob": None,
                                        "BiLSTM_RDKIT_label": None,
                                        "Consensus_Label": "Invalid_SMILES",
                                        "Error": err,
                                    }
                                )
                                continue

                            # Valid features
                            model_results = run_all_models_on_features(models, features)
                            consensus = consensus_from_results(model_results)

                            # Map per model to columns
                            row_result = {
                                "Input_SMILES": raw_smiles,
                                "Canonical_SMILES": features["canonical"],
                                "XGB_ECFP_prob": None,
                                "XGB_ECFP_label": None,
                                "XGB_MACCS_prob": None,
                                "XGB_MACCS_label": None,
                                "XGB_RDKIT_prob": None,
                                "XGB_RDKIT_label": None,
                                "BiLSTM_ECFP_prob": None,
                                "BiLSTM_ECFP_label": None,
                                "BiLSTM_MACCS_prob": None,
                                "BiLSTM_MACCS_label": None,
                                "BiLSTM_RDKIT_prob": None,
                                "BiLSTM_RDKIT_label": None,
                                "Consensus_Label": consensus,
                                "Error": None,
                            }

                            for r in model_results:
                                mname = r["Model"]
                                prob = r["Probability"]
                                dec = r["Decision"]

                                if mname == "XGB-ECFP":
                                    row_result["XGB_ECFP_prob"] = prob
                                    row_result["XGB_ECFP_label"] = dec
                                elif mname == "XGB-MACCS":
                                    row_result["XGB_MACCS_prob"] = prob
                                    row_result["XGB_MACCS_label"] = dec
                                elif mname == "XGB-RDKit":
                                    row_result["XGB_RDKIT_prob"] = prob
                                    row_result["XGB_RDKIT_label"] = dec
                                elif mname == "BiLSTM-ECFP":
                                    row_result["BiLSTM_ECFP_prob"] = prob
                                    row_result["BiLSTM_ECFP_label"] = dec
                                elif mname == "BiLSTM-MACCS":
                                    row_result["BiLSTM_MACCS_prob"] = prob
                                    row_result["BiLSTM_MACCS_label"] = dec
                                elif mname == "BiLSTM-RDKit":
                                    row_result["BiLSTM_RDKIT_prob"] = prob
                                    row_result["BiLSTM_RDKIT_label"] = dec

                            results_rows.append(row_result)

                        df_batch = pd.DataFrame(results_rows)

                        st.success("Batch prediction completed.")
                        st.subheader("Batch prediction results (first 20 rows)")
                        st.dataframe(df_batch.head(20))

                        csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download batch predictions as CSV",
                            data=csv_bytes,
                            file_name="CAIP_batch_predictions.csv",
                            mime="text/csv",
                        )

    # === Author Section ===
    st.markdown("---")
    st.subheader("About the Authors")

    col1, col2 = st.columns(2)

    with col1:
        try:
            image1 = Image.open("assets/duy.jpg")
            st.image(image1, caption="Huynh Anh Duy", width=160)
        except Exception:
            st.write("Huynh Anh Duy")
        st.markdown(
            """
**Huynh Anh Duy**  
*College of Natural Sciences, Can Tho University, Vietnam.  
**PhD Candidate, Faculty of Pharmaceutical Sciences,
Khon Kaen University, Thailand.  
*Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
"""
        )

    with col2:
        try:
            image2 = Image.open("assets/tarasi.png")
            st.image(image2, caption="Tarapong Srisongkram", width=160)
        except Exception:
            st.write("Asst Prof. Dr. Tarapong Srisongkram")
        st.markdown(
            """
**Asst Prof. Dr. Tarapong Srisongkram**  
*Faculty of Pharmaceutical Sciences,  
Khon Kaen University, Thailand.  
*Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
"""
        )

    # === Footer ===
    st.markdown("---")
    st.caption(
        f"Python {sys.version.split()[0]} • CAIP – Consensus Anti-inflammatory Predictor"
    )

    st.markdown(
    "<div style='text-align:center; font-size:14px; color:gray; line-height:1.6;'>"
    "⚠️ <b>Disclaimer:</b> This web application is intended for <i>**research use only**</i>. "
    "The information provided does not constitute and must not be used as a substitute for "
    "professional medical advice, diagnosis, or treatment. <br><br>"
    "🧪 Use of this tool is restricted to trained and qualified research personnel. <br><br>"
    "📄 <b>Version:</b> 1.0.0 &nbsp; | &nbsp; <b>Created on:</b> Jan 01, 2026 <br>"
    "© 2026 QSAR Lab &nbsp; | &nbsp; All rights reserved."
    "</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()
