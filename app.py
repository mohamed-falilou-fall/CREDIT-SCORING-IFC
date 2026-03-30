# -*- coding: utf-8 -*-
"""app - IFC AI Credit Scoring SaaS (version α)"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import shap
import matplotlib.pyplot as plt

# Utils
from utils.preprocessing import preprocess

# Agents
from agents.risk_agent import risk_analysis
from agents.financial_agent import financial_analysis
from agents.strategy_agent import strategy_analysis, train_strategy_model
from agents.decision_agent import final_decision
from agents.chat_agent import chat_ifc
from llm.rag_engine import build_vectorstore

# ================================
# CONFIG APP
# ================================
st.set_page_config(
    page_title="IFC AI Credit SaaS",
    layout="wide"
)

# ================================
# CONFIG APP
# ================================
st.set_page_config(
    page_title="IFC AI Credit SaaS",
    layout="wide"
)

# AJOUT DU FOND D'IMAGE
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Key South Lab - Team Epsilon - IFC AI Credit Scoring SaaS Platform (version α, nombre d’itérations réduit de 90 % pour des tests rapides sur Streamlit)")
st.markdown("**Mohamed Falilou Fall – Système d’agent IA Epsilon pour la prise de décision en matière de crédit (aligné sur les standards de l’International Finance Corporation et basé sur les six principales actions ainsi que sur des simulations pour McKinsey & Company)**")

# ================================
# SIDEBAR CONFIGURATION
# ================================
st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "1️. Choisir le modèle principal",
    ["Auto (Best)", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
)

run_ai = st.sidebar.checkbox("2️. Activer Multi Epsilon-Agent IA", value=True)

# ================================
# UPLOAD DATA
# ================================
st.header("Step 1: Charger le dataset")
uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # ================================
    # FILTRES CLIENT
    # ================================
    st.sidebar.header("3️. Filtrage des clients")

    if "pays_implantation" in df.columns:
        pays_list = ["Tous"] + sorted(df["pays_implantation"].dropna().unique().tolist())
        selected_pays = st.sidebar.selectbox("Pays d’implantation", pays_list)
    else:
        selected_pays = "Tous"
        st.sidebar.warning("Colonne 'pays_implantation' absente dans le dataset")

    if "secteur_activite" in df.columns:
        secteur_list = ["Tous"] + sorted(df["secteur_activite"].dropna().unique().tolist())
        selected_secteur = st.sidebar.selectbox("Secteur d'activité", secteur_list)
    else:
        selected_secteur = "Tous"
        st.sidebar.warning("Colonne 'secteur_activite' absente dans le dataset")

    id_input = st.sidebar.text_input("Rechercher par ID client")

    df_filtered = df.copy()
    if selected_pays != "Tous" and "pays_implantation" in df.columns:
        df_filtered = df_filtered[df_filtered["pays_implantation"] == selected_pays]
    if selected_secteur != "Tous" and "secteur_activite" in df.columns:
        df_filtered = df_filtered[df_filtered["secteur_activite"] == selected_secteur]
    if id_input and "id_client" in df.columns:
        df_filtered = df_filtered[df_filtered["id_client"].astype(str) == id_input]

    st.subheader("Données filtrées")
    st.dataframe(df_filtered)

    if df_filtered.empty:
        st.warning("Aucun client ne correspond aux filtres sélectionnés.")
        st.stop()

    # ================================
    # PREPROCESSING
    # ================================
    st.header("Step 2: Préprocessing")
    X, y = preprocess(df_filtered)

    if X is None or len(X) == 0:
        st.error("Dataset invalide après preprocessing")
        st.stop()

    st.success(f" Dataset prêt : {X.shape[0]} lignes, {X.shape[1]} variables")

    # ================================
    # TRAIN STRATEGY AGENT
    # ================================
    train_strategy_model(X)
    st.write(" Features utilisées :", list(X.columns))

    # ================================
    # MODELING
    # ================================
    st.header("Step 3: Modélisation")
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=50, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=50, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=50, verbose=0, random_state=42)
    }

    results = {}
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(mean_squared_error(y_val, preds))
            except:
                scores.append(np.nan)
        results[name] = np.nanmean(scores)

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE']).sort_values(by='MSE')
    st.subheader("Benchmark modèles")
    st.dataframe(results_df)

    best_model_name = results_df.index[0] if model_choice == "Auto (Best)" else model_choice
    best_model = models[best_model_name]
    st.success(f" Modèle sélectionné : {best_model_name}")

    best_model.fit(X, y)

    # ================================
    # SHAP
    # ================================
    st.header("Step 4: Explicabilité SHAP")
    try:
        with st.spinner("Calcul SHAP en cours..."):
            explainer = shap.Explainer(best_model, X)
            shap_values = explainer(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP non supporté pour ce modèle : {e}")

    # ================================
    # STEP 4B: RECOMMANDATIONS
    # ================================
    st.header("Step 4b: Diagnostic avancé et recommandations stratégiques")

    if 'shap_values' in locals():

        df_shap = pd.DataFrame(shap_values.values, columns=X.columns)
        features_analysis = [col for col in X.columns if col != "id_client"][:6]

        df_reco = []
        df_summary = []

        for i, row in df_shap.iterrows():

            client_id = df_filtered.iloc[i]["id_client"] if "id_client" in df_filtered.columns else i
            score_actuel = y.iloc[i]

            row_filtered = row[features_analysis]
            impacts = []
            score_gain_potentiel = 0

            for feat, val in row_filtered.items():

                if feat in df_filtered.columns:
                    valeur_client = df_filtered.iloc[i][feat]
                    moyenne = df_filtered[feat].mean()
                elif feat in X.columns:
                    valeur_client = X.iloc[i][feat]
                    moyenne = X[feat].mean()
                else:
                    valeur_client = np.nan
                    moyenne = np.nan

                gain = abs(val)
                score_gain_potentiel += gain

                impacts.append({
                    "feature": feat,
                    "impact": val,
                    "valeur": valeur_client,
                    "moyenne": moyenne,
                    "gain": gain
                })

            # Trier toutes les features par impact absolu et prendre top 6
            impacts_sorted = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)
            top6 = impacts_sorted[:6]

            recommandations_top6 = []
            for t in top6:
                if pd.notna(t["valeur"]) and pd.notna(t["moyenne"]):
                    direction = "augmenter" if t["valeur"] < t["moyenne"] else "optimiser"
                else:
                    direction = "améliorer"

                recommandations_top6.append(
                    f"{t['feature']} ({direction}, impact={round(t['impact'],3)})"
                )

            score_simule = min(1.0, score_actuel + score_gain_potentiel)

            if score_actuel < 0.4:
                segment = "High Risk"
            elif score_actuel < 0.7:
                segment = "Turnaround"
            else:
                segment = "Strong"

            df_summary.append({
                "id_client": client_id,
                "score_actuel": round(score_actuel, 4),
                "score_potentiel": round(score_simule, 4),
                "gain_potentiel": round(score_gain_potentiel, 4),
                "segment": segment,
                "top_6_actions": " | ".join(recommandations_top6) if recommandations_top6 else "RAS"
            })

            # Construire df_reco pour ces top6 seulement
            for t in top6:
                df_reco.append({
                    "id_client": client_id,
                    "feature_critique": t["feature"],
                    "impact_shap": round(t["impact"], 4),
                    "valeur_client": t["valeur"],
                    "benchmark_moyenne": round(t["moyenne"], 4) if pd.notna(t["moyenne"]) else np.nan,
                    "gain_potentiel": round(t["gain"], 4)
                })

        df_reco = pd.DataFrame(df_reco)
        df_summary = pd.DataFrame(df_summary)

        st.subheader("Résumé exécutif par client (Top 6 actions & simulation)")
        st.dataframe(df_summary.sort_values("score_actuel", ascending=False))

        st.subheader("Analyse détaillée")
        st.dataframe(df_reco.sort_values(["id_client", "impact_shap"]))

    else:
        st.warning("Les valeurs SHAP ne sont pas disponibles.")

    # ================================
    # EXPECTED LOSS
    # ================================
    st.header("Step 5: Expected Loss")
    if all(col in df_filtered.columns for col in ["probabilite_defaut", "perte_en_cas_defaut", "exposition_defaut"]):
        df_filtered["expected_loss"] = (
            df_filtered["probabilite_defaut"]
            * df_filtered["perte_en_cas_defaut"]
            * df_filtered["exposition_defaut"]
        )
        st.dataframe(df_filtered[["expected_loss"]].head())
    else:
        st.warning("Colonnes nécessaires pour Expected Loss manquantes.")

    # ================================
    # EPSILON AGENT AI
    # ================================
    if run_ai:
        st.header("Step 6: Epsilon-Agent AI System")

        index = st.slider("Sélectionner un client", 0, len(X) - 1)

        if st.button("Lancer analyse IA"):
            row_data = X.iloc[index].to_dict()
            shap_row = shap_values[index].values.tolist() if 'shap_values' in locals() else None
            expected_loss = df_filtered.iloc[index]["expected_loss"] if "expected_loss" in df_filtered.columns else 0

            with st.spinner("Analyse IA en cours..."):
                risk = risk_analysis(row_data, shap_row)
                financial = financial_analysis(expected_loss)
                strategy = strategy_analysis(row_data)
                decision = final_decision(risk, financial, strategy)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Risk Agent")
                st.write(risk)
                st.subheader("Financial Agent")
                st.write(financial)
            with col2:
                st.subheader("Strategy Agent")
                st.write(strategy)
                st.subheader("Decision Agent")
                st.success(decision)

    # ================================
    # RAG IFC
    # ================================
    st.sidebar.header("7️. LLM IFC - RAG")
    if st.sidebar.button("Indexer les rapports IFC (PDF)"):
        with st.spinner("Indexation en cours..."):
            build_vectorstore("report/")
        st.success("Vectorstore IFC prêt")

    st.header("Step 7: IFC AI Chat")
    user_question = st.text_input("Pose une question sur les données ou le modèle")

    if user_question:
        context = f"""
Dataset summary:
{X.describe().to_string()}

Model performance:
{results_df.to_string()}
"""
        response = chat_ifc(context, user_question)
        st.write(response)

else:
    st.info("")
