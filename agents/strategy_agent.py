# -*- coding: utf-8 -*-
"""
strategy_agent.py

Analyse stratégique IFC
Mode test sans clé OpenAI pour déploiement Streamlit Cloud.
"""

import random

def strategy_analysis(ca_prediction, risk_score):
    """
    Analyse stratégique du client IFC.
     Mode test : aucun appel OpenAI si pas de clé
    """

    try:
        import streamlit as st
        from openai import OpenAI

        # -------------------------------
        # Mode OpenAI (optionnel)
        # -------------------------------
        if "OPENAI_API_KEY" in st.secrets:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            prompt = f"""
            Tu es expert IFC en stratégie.

            Prévisions de CA : {ca_prediction}
            Score de risque : {risk_score}

            Donne :
            - Recommandation stratégique
            - Opportunités
            - Points d'attention
            """

            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            return response.choices[0].message.content

        else:
            raise ValueError("Pas de clé API, mode test activé")

    except:
        # -------------------------------
        # MODE TEST (Simulation)
        # -------------------------------
        recommandations = ["Renforcer capital", "Investissement modéré", "Pas d'action immédiate"]
        opportunites = ["Expansion marché", "Diversification produit", "Partenariat stratégique"]
        points_attention = ["Risque secteur", "Exposition client élevée", "Cash-flow limité"]

        return f"""
         MODE TEST (sans OpenAI)

         CA prévisionnel : {ca_prediction}
         Score de risque : {risk_score}
         Recommandation stratégique : {random.choice(recommandations)}
         Opportunités : {random.choice(opportunites)}
         Points d'attention : {random.choice(points_attention)}
        """
