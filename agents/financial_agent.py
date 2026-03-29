# -*- coding: utf-8 -*-
"""
financial_agent.py

Analyse financière du client IFC
Mode test sans clé OpenAI pour déploiement Streamlit Cloud.
"""

import random

def financial_analysis(expected_loss):
    """
    Analyse financière du client IFC.
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
            Tu es analyste financier IFC.

            Expected Loss : {expected_loss}

            Donne :
            - Niveau de perte
            - Soutenabilité
            - Risque financier
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
        niveau = "Faible" if expected_loss < 10000 else "Modéré" if expected_loss < 50000 else "Élevé"
        soutenabilite = "OK" if expected_loss < 20000 else "Attention" 
        risque = "" if expected_loss < 10000 else "" if expected_loss < 50000 else "🔴"

        return f"""
         MODE TEST (sans OpenAI)

         Expected Loss : {expected_loss}
         Niveau de perte : {niveau}
         Soutenabilité : {soutenabilite}
         Risque financier : {risque}
        """
