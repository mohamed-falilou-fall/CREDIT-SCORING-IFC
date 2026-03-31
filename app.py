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

st.markdown("")

# ================================
# BACKGROUND
# ================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/071/848/200/small/flat-design-world-globe-grid-icon-symbol-sign-illustration-graphic-png.png");
        background-size: 660px;  /* */
        background-position: center center;  /*  */
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# ================================
# HEADER
# ================================
st.title("WORLD BANK GROUP AI Credit Scoring SaaS Platform (version α, nombre d’itérations réduit de 90 % pour des tests rapides sur Streamlit)")

st.markdown("""
### Plateforme d’aide à la décision crédit basée sur l’IA

Cette application de Mohamed Falilou Fall permet d’analyser des portefeuilles clients, d’évaluer le risque de crédit, 
et de générer des recommandations stratégiques automatisées alignées sur les standards de la Banque Mondiale (IFC, World Bank, MIGA).

**Fonctionnalités principales :**
- Analyse de risque multi-modèles
- Explicabilité avancée SHAP(Shapley Additive Explanations), méthode qui permet de comprendre précisément pourquoi un modèle donne un score donné
- Simulation d’amélioration du score crédit
- Système multi-agents IA (Epsilon)
- Chat intelligent basé sur documents IFC (RAG)

---
""")

st.markdown("**Objectif métier :** reproduire une approche de type IFC / Banque mondiale / McKinsey & Company combinant scoring quantitatif, explicabilité et recommandations stratégiques automatisées.")

# ================================
# SIDEBAR CONFIGURATION
# ================================
st.sidebar.header("Configuration générale")

st.sidebar.markdown("""
Personnalisez ici le comportement du moteur d’analyse :

- Choix du modèle IA
- Activation du système multi-agents
""")

st.sidebar.markdown("**Paramétrage utilisateur :** cette section permet d’ajuster dynamiquement le moteur de scoring et d’activer l’intelligence multi-agents.")

model_choice = st.sidebar.selectbox(
    "1️. Choisir le modèle principal (moteur de scoring)",
    ["Auto (Best)", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
)

run_ai = st.sidebar.checkbox(
    "2️. Activer le système Multi Epsilon-Agent IA",
    value=True,
    help="Active l’analyse automatique par agents spécialisés (risque, finance, stratégie, décision)"
)

# ================================
# UPLOAD DATA
# ================================
st.header("Step 1: Chargement des données")

st.markdown("""
Chargez un fichier CSV contenant les informations clients.

Le dataset doit idéalement inclure :
- Variables financières
- Informations sectorielles
- Identifiant client
""")

st.markdown("**Entrée des données :** cette étape constitue la base du pipeline. La qualité des analyses dépend directement de la qualité du dataset importé.")

uploaded_file = st.file_uploader("Importer votre dataset CSV", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu du dataset")
    st.markdown("Visualisation rapide des premières lignes du fichier importé.")
    st.dataframe(df.head())

    st.markdown("**Exploration initiale :** permet de vérifier rapidement la structure, les variables disponibles et détecter d’éventuelles anomalies.")

    # ================================
    # FILTRES CLIENT
    # ================================
    st.sidebar.header("3️. Filtrage des clients")

    st.sidebar.markdown("""
Affinez votre analyse en sélectionnant un sous-ensemble de clients :
- Par pays
- Par secteur
- Par identifiant spécifique
""")

    st.sidebar.markdown("**Segmentation analytique :** permet de cibler des sous-portefeuilles pour une analyse plus précise (ex : secteur, pays, client spécifique).")

    if "pays_implantation" in df.columns:
        pays_list = ["Tous"] + sorted(df["pays_implantation"].dropna().unique().tolist())
        selected_pays = st.sidebar.selectbox("Pays d’implantation", pays_list)
    else:
        selected_pays = "Tous"
        st.sidebar.warning("Colonne 'pays_implantation' absente")

    if "secteur_activite" in df.columns:
        secteur_list = ["Tous"] + sorted(df["secteur_activite"].dropna().unique().tolist())
        selected_secteur = st.sidebar.selectbox("Secteur d'activité", secteur_list)
    else:
        selected_secteur = "Tous"
        st.sidebar.warning("Colonne 'secteur_activite' absente")

    id_input = st.sidebar.text_input("Rechercher un client par ID")

    df_filtered = df.copy()

    if selected_pays != "Tous" and "pays_implantation" in df.columns:
        df_filtered = df_filtered[df_filtered["pays_implantation"] == selected_pays]

    if selected_secteur != "Tous" and "secteur_activite" in df.columns:
        df_filtered = df_filtered[df_filtered["secteur_activite"] == selected_secteur]

    if id_input and "id_client" in df.columns:
        df_filtered = df_filtered[df_filtered["id_client"].astype(str) == id_input]

    st.subheader("Données filtrées")
    st.dataframe(df_filtered)

    st.markdown("**Dataset final utilisé :** correspond au périmètre exact sur lequel seront effectuées toutes les analyses suivantes.")

    if df_filtered.empty:
        st.warning("Aucun client ne correspond aux critères sélectionnés.")
        st.stop()

    # ================================
    # PREPROCESSING
    # ================================
    st.header("Step 2: Préparation des données")

    st.markdown("""
Transformation automatique des données :
- Nettoyage
- Encodage
- Sélection des variables
""")

    st.markdown("**Data engineering :** transformation du dataset brut en matrice exploitable par les modèles (feature engineering simplifié).")

    X, y = preprocess(df_filtered)

    if X is None or len(X) == 0:
        st.error("Dataset invalide après preprocessing")
        st.stop()

    st.success(f"Dataset prêt : {X.shape[0]} observations, {X.shape[1]} variables")

    train_strategy_model(X)

    st.write("Variables utilisées :", list(X.columns))

    # ================================
    # MODELING
    # ================================
    st.header("Step 3: Modélisation & Benchmark")

    st.markdown("""
Comparaison automatique de plusieurs modèles de Machine Learning 
afin de sélectionner le plus performant.
""")

    st.markdown("**Benchmark IA :** évaluation comparative des modèles pour sélectionner celui qui minimise l’erreur de prédiction (MSE : Erreur Quadradique Moyenne).")

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

    st.subheader("Résultats des modèles")
    st.dataframe(results_df)

    best_model_name = results_df.index[0] if model_choice == "Auto (Best)" else model_choice
    best_model = models[best_model_name]

    st.success(f"Modèle sélectionné : {best_model_name}")

    best_model.fit(X, y)

    # ================================
    # SHAP
    # ================================
    st.header("Step 4: Explicabilité du modèle (SHAP)")

    st.markdown("""
Analyse des variables influençant les décisions du modèle.
""")

    st.markdown("**Explainable AI (XAI) :** SHAP (Shapley Additive Explanations) est la méthode qui permet de comprendre précisément pourquoi un modèle donne un score donné (interprétabilité locale et globale).")

    try:
        with st.spinner("Calcul des contributions SHAP..."):
            explainer = shap.Explainer(best_model, X)
            shap_values = explainer(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP non supporté : {e}")

    # ================================
    # STEP 4B: RECOMMANDATIONS
    # ================================
    st.header("Step 4b: Diagnostic avancé et recommandations stratégiques")

    st.markdown("**Moteur de recommandation :** transforme les insights SHAP en actions concrètes pour améliorer le score crédit.")

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

    st.markdown("**Risk management financier :** calcul de la perte attendue selon la formule IFC (PD × LGD × EAD).")

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

        st.markdown("**Architecture multi-agents :** simulation d’un comité de crédit intelligent (risque, finance, stratégie, décision).")

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

    st.sidebar.markdown("**Intelligence documentaire :** permet d’interroger des rapports IFC via un moteur RAG (Retrieval-Augmented Generation).")

    if st.sidebar.button("Indexer les rapports IFC (PDF)"):
        with st.spinner("Indexation en cours..."):
            build_vectorstore("report/")
        st.success("Vectorstore IFC prêt")

    st.header("Step 7: IFC AI Chat")

    st.markdown("**Assistant intelligent :** permet de poser des questions sur les données, les modèles ou les standards IFC.")

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
