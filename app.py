# streamlit_app_ifc_final_no_tabnet.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="IFC Credit Scoring", layout="wide")
st.title("Key South Lab - Team Εpsilon - IFC Credit Scoring Dashboard (version α, nombre d’itérations réduit de 90 % pour des tests rapides sur Streamlit)")

st.markdown("**Mohamed Falilou Fall**")

# ================================
# 1️ Upload du dataset
# ================================
st.header("1. Dataset")
uploaded_file = st.file_uploader("Charger le dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu du dataset :", df.head())

    # ================================
    # 2️ Préprocessing
    # ================================
    st.header("2. Préprocessing")
    X = df.drop(columns=["probabilite_defaut"])
    y = df["probabilite_defaut"]

    X_clean = X.select_dtypes(include=['number'])
    if "classe_risque" in X_clean.columns:
        X_clean = X_clean.drop(columns=["classe_risque"])

    mask = ~y.isna()
    X_clean = X_clean[mask]
    y_clean = y[mask]
    X_clean = X_clean.fillna(0)

    # Feature engineering simple
    if "score_risque_pays" in X_clean.columns:
        X_clean["mean_risk_country"] = X_clean.groupby("score_risque_pays")["score_risque_pays"].transform("mean")
    if "croissance_pays" in X_clean.columns and "risque_sectoriel" in X_clean.columns:
        X_clean["interaction_sector_macro"] = X_clean["croissance_pays"] * X_clean["risque_sectoriel"]

    st.success("Préprocessing terminé.")

    # ================================
    # 3️ Modélisation
    # ================================
    st.header("3. Modélisation")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=50, random_state=42),
        "HistGB": HistGradientBoostingRegressor(max_iter=50, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=50, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=50, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=50, verbose=0, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(128,64), max_iter=100, random_state=42),
        "FT-Transformer (proxy)": MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=100)
    }

    # ================================
    # 4️ Benchmark des modèles
    # ================================
    st.header("4. Benchmark des modèles")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        st.write(f"🔹 Entraînement du modèle : {name}")
        scores = []

        for train_idx, val_idx in kf.split(X_clean):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(mean_squared_error(y_val, preds))
            except Exception as e:
                st.warning(f"Erreur avec {name} : {e}")
                scores.append(np.nan)

        results[name] = np.nanmean(scores)

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE']).sort_values(by='MSE')
    st.dataframe(results_df)

    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    st.success(f"Meilleur modèle : {best_model_name}")

    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(results_df.index, results_df["MSE"], color=plt.cm.viridis(np.linspace(0,1,len(results_df))))
    for bar, mse in zip(bars, results_df["MSE"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{mse:.4f}", ha='center', fontsize=9)
    ax.set_xticklabels(results_df.index, rotation=45)
    ax.set_ylabel("Erreur MSE")
    ax.set_title("Comparaison des modèles (MSE)")
    st.pyplot(fig)

    # ================================
    # 5️ Score et décision
    # ================================
    st.header("5. Score et décision")
    df["score_ifc"] = df.get("score_ifc", np.random.rand(len(df)))
    df["decision"] = df["score_ifc"].apply(lambda x: "Investir" if x>0 else ("Conditionnel" if x==0 else "Refuser"))
    st.dataframe(df[["score_ifc","decision"]].head())

    # ================================
    # 6️ Expected Loss
    # ================================
    st.header("6. Expected Loss")
    df["expected_loss"] = df["probabilite_defaut"] * df["perte_en_cas_defaut"] * df["exposition_defaut"]
    st.dataframe(df[["expected_loss"]].head())

    # ================================
    # 7️ Explicabilité SHAP
    # ================================
    st.header("7. Explicabilité SHAP")
    explainer = shap.Explainer(best_model, X_clean)
    shap_values = explainer(X_clean)

    fig, ax = plt.subplots(figsize=(10,6))
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)

else:
    st.info("En attente de chargement du dataset pour exécuter le pipeline.")
