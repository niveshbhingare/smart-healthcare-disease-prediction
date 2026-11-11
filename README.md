# Smart Healthcare Data Analytics for Disease Prediction

Assignment project: train a model to predict a disease **category** from
symptoms, and serve it through an interactive dashboard.

## Files

- `disease_dataset.csv` — the dataset you provided (100,000 rows, 174 binary
  symptom columns, 46 individual diseases).
- `train_model.py` — maps the 46 diseases into 10 broad body-system
  categories, trains and evaluates a model, saves it to `./model/`.
- `dashboard_app.py` — Streamlit dashboard for interactive prediction + EDA.
- `requirements.txt` — Python packages needed.

## How to run

1. Install dependencies (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (creates a `model/` folder with the saved model + metrics):
   ```bash
   python train_model.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run dashboard_app.py
   ```
   Opens at `http://localhost:8501`.

## Why predict categories instead of all 46 diseases?

This is the most important design decision in the project, and it's worth
explaining clearly in your report.

**The raw symptom data is genuinely noisy.** I checked this directly: for
Asthma (21,295 rows — the single largest disease class), the *most*
characteristic symptom in the dataset is still only present in 36% of
Asthma cases. Most symptom columns hover between 30–50% prevalence
regardless of which disease you look at. That means the class-conditional
distributions overlap heavily — there just isn't a strong learnable signal
tying most symptoms to a specific one of 46 diseases. A model trained on
all 46 individual diseases tops out around **35–36% test accuracy**, no
matter which algorithm is used (verified with both Random Forest and
Logistic Regression; more complex models like XGBoost won't do meaningfully
better because the ceiling is set by the data, not the model).

**Grouping into 10 body-system categories** (Respiratory, Cardiovascular,
Gastrointestinal, Endocrine/Metabolic, Neurological, Renal/Urinary,
Musculoskeletal/Connective, Mental Health, Infectious Disease, Autoimmune/
Genetic/Rare) is a legitimate way to raise real accuracy: symptoms are far
more genuinely predictive of *which body system* is affected than of the
exact disease. This lifts accuracy to about **46–47%**, and — importantly —
that's a real ~9-point improvement over the "always guess the biggest
category" baseline of 37.7%, not just an artifact of fewer classes.

### A trap to avoid: don't chase 80% by trivializing the problem

If you merge categories further (e.g. down to 3–4 groups, or a single
"Asthma vs. everything else" binary problem), the accuracy number climbs —
but mostly because the majority class baseline climbs with it. For example,
a binary "Asthma vs. not Asthma" classifier scores ~84%, but simply *always*
guessing "not Asthma" already scores ~79% due to class imbalance — so the
model is barely adding anything. If your report includes an accuracy number,
also report the majority-class baseline next to it, and use the **lift over
baseline** as the real measure of model skill. `model_metrics.json` (produced
by `train_model.py`) includes `baseline_accuracy` for exactly this reason,
and the dashboard's "Model Performance" tab surfaces it.

| Granularity | Model accuracy | Majority-class baseline | Real lift |
|---|---|---|---|
| 46 individual diseases | ~36% | ~21% | +15 pts |
| **10 categories (used here)** | **~47%** | **~38%** | **+9 pts** |
| 4 categories | ~67% | ~59% | +8 pts |
| Binary (Asthma vs. rest) | ~84% | ~79% | +5 pts |

## What the model does

- **Input**: 174 binary symptom flags (0/1).
- **Output**: predicted disease category out of 10 classes, with a
  probability score, plus a secondary "closest specific disease" hint
  within that category (based on symptom-profile similarity, not a
  separate model).
- **Algorithm**: Random Forest Classifier (compared against a Logistic
  Regression baseline; the better one on test accuracy is saved
  automatically).

## Ideas for extra credit / discussion in your report

- Discuss *why* the ceiling exists: this looks like a synthetically
  generated dataset with substantial injected noise, which is realistic —
  real-world symptom-only diagnosis is genuinely hard, and this project
  demonstrates that limitation rather than hiding it.
- Try gradient boosting (XGBoost/LightGBM) and show it doesn't meaningfully
  beat Random Forest here — reinforces that the bottleneck is data, not
  algorithm choice.
- Add SHAP values to the dashboard for per-prediction symptom explanations.
- Discuss what additional data (lab results, vital signs, patient history)
  would likely be needed to push real-world accuracy higher.
