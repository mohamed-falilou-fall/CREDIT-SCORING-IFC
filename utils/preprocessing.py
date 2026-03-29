# -*- coding: utf-8 -*-
"""
preprocessing.py
Préprocessing des données IFC
"""

import pandas as pd
import numpy as np

def preprocess(df):
    X = df.drop(columns=["probabilite_defaut"], errors='ignore')
    y = df["probabilite_defaut"] if "probabilite_defaut" in df.columns else pd.Series(np.zeros(len(df)))

    X = X.select_dtypes(include=['number'])

    if "classe_risque" in X.columns:
        X = X.drop(columns=["classe_risque"])

    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    X = X.fillna(0)

    # Feature engineering
    if "score_risque_pays" in X.columns:
        X["mean_risk_country"] = X.groupby("score_risque_pays")["score_risque_pays"].transform("mean")

    if "croissance_pays" in X.columns and "risque_sectoriel" in X.columns:
        X["interaction_sector_macro"] = X["croissance_pays"] * X["risque_sectoriel"]

    return X, y
