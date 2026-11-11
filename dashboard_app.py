"""
dashboard_app.py
------------------
Streamlit dashboard for the Smart Healthcare Data Analytics project.

Predicts a broad disease CATEGORY (body system) from selected symptoms,
and additionally shows which specific diseases within that category are
most consistent with the symptoms (by comparing to average symptom
profiles in the training data) as a secondary, more granular hint.

Run:
    streamlit run dashboard_app.py

Requires that train_model.py has already been run once, so that the
./model/ folder contains disease_model.pkl, label_encoder.pkl,
feature_columns.pkl, category_map.pkl and model_metrics.json.
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_DIR = "model"
DATA_PATH = "disease_dataset.csv"

st.set_page_config(
    page_title="Smart Healthcare Disease Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    category_map = joblib.load(os.path.join(MODEL_DIR, "category_map.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)
    return model, le, feature_cols, category_map, metrics


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def disease_symptom_profiles(_feature_cols):
    """Average symptom prevalence per individual disease, for the
    within-category 'closest match' hint on the Predict tab."""
    df = load_dataset()
    feature_cols_list = list(_feature_cols)
    return df.groupby("disease")[feature_cols_list].mean()


def prettify(col_name: str) -> str:
    name = col_name.replace("symptom_", "").replace("rare_", "rare: ")
    return name.replace("_", " ").strip().capitalize()


def group_features(feature_cols):
    """Split symptom columns into a few readable groups for the sidebar."""
    groups = {
        "Respiratory": [], "Cardiac": [], "Neurological": [], "Digestive": [],
        "General / Systemic": [], "Musculoskeletal & Skin": [], "Mental Health": [],
        "Rare / Genetic": [], "Other": [],
    }
    resp = ["cough", "breath", "wheez", "chest_tightness", "sinus", "nasal", "throat", "smell", "taste", "sputum", "hemoptysis"]
    card = ["chest_pain", "palpit", "tachy", "brady", "cyanosis", "edema", "claudication", "syncope", "clammy"]
    neuro = ["headache", "migraine", "dizz", "vertigo", "seizure", "weakness", "numbness", "tingling", "vision", "speech", "confusion", "memory", "tremor", "gait", "ataxia", "paresthesia", "facial_droop", "photophobia", "phonophobia"]
    digest = ["nausea", "vomit", "diarrhea", "constipation", "abdominal", "heartburn", "appetite", "gi_bleeding", "stool", "jaundice", "dysphagia", "satiety", "hematemesis", "melena", "bloating", "regurgitation"]
    general = ["fatigue", "fever", "chills", "sweat", "weight", "polyuria", "polydipsia", "polyphagia", "intolerance", "hair_loss"]
    msk_skin = ["joint", "muscle", "myalgia", "arthralgia", "back_pain", "neck_pain", "stiffness", "rash", "urticaria", "bruis", "petechiae", "skin", "puffiness", "pruritus", "bone", "limp", "tendon"]
    mental = ["mood", "anhedonia", "insomnia", "hypersomnia", "anxiety", "panic", "obsession", "compulsion", "withdrawal", "suicidal"]

    for col in feature_cols:
        lc = col.lower()
        if col.startswith("rare_symptom"):
            groups["Rare / Genetic"].append(col)
        elif any(k in lc for k in resp):
            groups["Respiratory"].append(col)
        elif any(k in lc for k in card):
            groups["Cardiac"].append(col)
        elif any(k in lc for k in neuro):
            groups["Neurological"].append(col)
        elif any(k in lc for k in digest):
            groups["Digestive"].append(col)
        elif any(k in lc for k in mental):
            groups["Mental Health"].append(col)
        elif any(k in lc for k in msk_skin):
            groups["Musculoskeletal & Skin"].append(col)
        elif any(k in lc for k in general):
            groups["General / Systemic"].append(col)
        else:
            groups["Other"].append(col)
    return {k: v for k, v in groups.items() if v}


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

if not os.path.exists(os.path.join(MODEL_DIR, "disease_model.pkl")):
    st.error("Model not found. Please run `python train_model.py` first to train and save the model.")
    st.stop()

model, le, feature_cols, category_map, metrics = load_model()

st.title("🩺 Smart Healthcare Disease Predictor")
st.caption(
    f"Data analytics dashboard — select symptoms, and the trained {metrics['model_name']} "
    "model predicts the most likely disease **category** (body system)."
)

tab_predict, tab_data, tab_model = st.tabs(["🔍 Predict", "📊 Dataset Overview", "📈 Model Performance"])

# ---------------------------------------------------------------------------
# TAB 1: Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Select symptoms")
    st.write("Tick every symptom that applies, then click **Predict**.")

    grouped = group_features(feature_cols)
    selected = set()

    cols = st.columns(2)
    group_names = list(grouped.keys())
    for i, group_name in enumerate(group_names):
        with cols[i % 2]:
            with st.expander(group_name, expanded=(i < 2)):
                for feat in grouped[group_name]:
                    if st.checkbox(prettify(feat), key=feat):
                        selected.add(feat)

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
        st.metric("Symptoms selected", len(selected))

    with right:
        if predict_clicked:
            if not selected:
                st.warning("Please select at least one symptom before predicting.")
            else:
                input_row = pd.DataFrame(
                    [[1 if f in selected else 0 for f in feature_cols]], columns=feature_cols
                )
                probs = model.predict_proba(input_row)[0]
                pred_idx = probs.argmax()
                pred_category = le.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx]

                st.success(f"### Predicted category: **{pred_category.replace('_', ' ')}**")
                st.write(f"Model confidence: **{confidence:.1%}**")

                top_n = 5
                top_idx = probs.argsort()[::-1][:top_n]
                top_df = pd.DataFrame(
                    {
                        "Category": [le.classes_[i].replace("_", " ") for i in top_idx],
                        "Probability": [probs[i] for i in top_idx],
                    }
                )
                fig = px.bar(
                    top_df.sort_values("Probability"), x="Probability", y="Category",
                    orientation="h", title="Top categories by probability",
                    color="Probability", color_continuous_scale="Tealgrn",
                )
                fig.update_layout(yaxis_title="", xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

                # Secondary hint: closest specific disease(s) within the predicted category
                st.markdown("#### Closest specific diseases within this category")
                st.caption(
                    "This compares your selected symptoms to the average symptom profile "
                    "of each disease in the predicted category. It's a similarity hint, "
                    "not a separate model prediction."
                )
                profiles = disease_symptom_profiles(tuple(feature_cols))
                diseases_in_cat = [d for d, c in category_map.items() if c == pred_category]
                sub_profiles = profiles.loc[profiles.index.intersection(diseases_in_cat)]
                sel_vector = pd.Series({f: (1 if f in selected else 0) for f in feature_cols})
                # cosine-style similarity between selected symptoms and each disease's average profile
                sims = sub_profiles.apply(lambda row: (row * sel_vector).sum() / (
                    (row.pow(2).sum() ** 0.5) * (sel_vector.pow(2).sum() ** 0.5) + 1e-9
                ), axis=1).sort_values(ascending=False)
                sim_df = sims.head(5).reset_index()
                sim_df.columns = ["Disease", "Similarity"]
                sim_df["Disease"] = sim_df["Disease"].str.replace("_", " ")
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                st.info(
                    "⚠️ This is an educational demo built on synthetic data for a "
                    "data-analytics assignment. It is **not** a medical diagnostic "
                    "tool and must not be used for real clinical decisions."
                )

# ---------------------------------------------------------------------------
# TAB 2: Dataset overview
# ---------------------------------------------------------------------------
with tab_data:
    df = load_dataset()
    df["category"] = df["disease"].map(category_map)

    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Symptom features", len(feature_cols))
    c3.metric("Categories (from 46 diseases)", df["category"].nunique())

    st.markdown("#### Category distribution")
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig1 = px.bar(counts, x="Count", y="Category", orientation="h", height=450)
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Individual diseases within each category")
    disease_counts = df.groupby(["category", "disease"]).size().reset_index(name="count")
    selected_cat = st.selectbox("Choose a category to expand", sorted(df["category"].unique()))
    sub = disease_counts[disease_counts["category"] == selected_cat].sort_values("count", ascending=True)
    sub["disease"] = sub["disease"].str.replace("_", " ")
    fig1b = px.bar(sub, x="count", y="disease", orientation="h")
    st.plotly_chart(fig1b, use_container_width=True)

    st.markdown("#### Most common symptoms overall")
    symptom_freq = df[feature_cols].mean().sort_values(ascending=False).head(20).reset_index()
    symptom_freq.columns = ["Symptom", "Frequency"]
    symptom_freq["Symptom"] = symptom_freq["Symptom"].apply(prettify)
    fig2 = px.bar(symptom_freq, x="Frequency", y="Symptom", orientation="h")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Explore a sample of the raw data")
    st.dataframe(df.sample(min(200, len(df)), random_state=1), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Model performance
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", metrics["model_name"])
    c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("Majority-class baseline", f"{metrics['baseline_accuracy']:.1%}")
    c4.metric("Real lift over baseline", f"+{(metrics['accuracy'] - metrics['baseline_accuracy']) * 100:.1f} pts")

    st.info(
        "Reporting accuracy alongside the 'always guess the biggest class' baseline "
        "matters here because the categories are imbalanced. The **lift over baseline** "
        "is what actually reflects the model's learned skill."
    )

    st.markdown("#### Random Forest vs Logistic Regression")
    comp = metrics["comparison"]
    comp_df = pd.DataFrame(
        {
            "Model": ["Random Forest", "Logistic Regression"],
            "Accuracy": [comp["random_forest"]["accuracy"], comp["logistic_regression"]["accuracy"]],
            "Macro F1": [comp["random_forest"]["macro_f1"], comp["logistic_regression"]["macro_f1"]],
        }
    )
    fig3 = px.bar(comp_df, x="Model", y=["Accuracy", "Macro F1"], barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    if metrics.get("top_features"):
        st.markdown("#### Top predictive symptoms (feature importance)")
        feat_df = pd.DataFrame(metrics["top_features"], columns=["Symptom", "Importance"])
        feat_df["Symptom"] = feat_df["Symptom"].apply(prettify)
        fig4 = px.bar(feat_df.sort_values("Importance"), x="Importance", y="Symptom", orientation="h")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Per-category performance")
    report = metrics["classification_report"]
    rows = []
    for cat, vals in report.items():
        if cat in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append(
            {
                "Category": cat.replace("_", " "),
                "Precision": vals["precision"],
                "Recall": vals["recall"],
                "F1-score": vals["f1-score"],
                "Support": int(vals["support"]),
            }
        )
    report_df = pd.DataFrame(rows).sort_values("Support", ascending=False)
    st.dataframe(report_df, use_container_width=True, height=400)

    st.caption(
        "Note: performance still varies by category because of imbalance (Respiratory "
        "has 37,725 samples; Infectious Disease has only 86). This is discussed in the README."
    )
