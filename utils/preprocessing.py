# -*- coding: utf-8 -*-
"""
preprocessing.py
Préprocessing des données IFC - Version avancée ML + Agents IA
"""

import pandas as pd
import numpy as np

def preprocess(df):

    # ================================
    # 1. TARGET
    # ================================
    y = df["probabilite_defaut"] if "probabilite_defaut" in df.columns else pd.Series(np.zeros(len(df)))

    # ================================
    # 2. FEATURES
    # ================================
    X = df.drop(columns=["probabilite_defaut"], errors='ignore')

    # Garde uniquement numérique
    X = X.select_dtypes(include=['number'])

    # Supprime variable fuite
    if "classe_risque" in X.columns:
        X = X.drop(columns=["classe_risque"])

    # ================================
    # 3. CLEANING
    # ================================
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    X = X.fillna(0)

    # ================================
    # 4. FEATURE ENGINEERING (CORE IFC)
    # ================================

    #  Ratio crédit / revenu (hyper important en scoring)
    if "credit" in X.columns and "revenu" in X.columns:
        X["ratio_credit_revenu"] = X["credit"] / (X["revenu"] + 1)

    #  Risque pays moyen
    if "score_risque_pays" in X.columns:
        X["mean_risk_country"] = X.groupby("score_risque_pays")["score_risque_pays"].transform("mean")

    #  Interaction macro-économique
    if "croissance_pays" in X.columns and "risque_sectoriel" in X.columns:
        X["interaction_sector_macro"] = X["croissance_pays"] * X["risque_sectoriel"]

    #  Approximation capacité remboursement
    if "revenu" in X.columns and "credit" in X.columns:
        X["capacite_remboursement"] = X["revenu"] - X["credit"]

    # ️ Score de stress financier
    if "ratio_credit_revenu" in X.columns:
        X["stress_financier"] = np.where(
            X["ratio_credit_revenu"] > 0.5, 1, 0
        )

    # ================================
    # 5. NORMALISATION (important pour clustering)
    # ================================
    for col in X.columns:
        if X[col].std() > 0:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-6)

    # ================================
    # 6. SÉCURITÉ
    # ================================
    X = X.replace([np.inf, -np.inf], 0)

    # ================================
    # 7. INFO DEBUG (optionnel)
    # ================================
    print(f"[INFO] Dataset preprocessé : {X.shape}")

    return X, y
