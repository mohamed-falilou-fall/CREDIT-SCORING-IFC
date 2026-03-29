# agents/risk_agent.py

import random

# -------------------------------
# 🔹 Fonction principale
# -------------------------------
def risk_analysis(data, shap_values=None):
    """
    Analyse du risque de crédit avec :
    - Mode API OpenAI (si clé dispo)
    - Mode fallback simulé (si pas de clé)

    :param data: dict contenant les infos client
    :param shap_values: dict (optionnel) importance variables
    :return: string analyse risque
    """

    # -------------------------------
    # 🔹 Tentative utilisation OpenAI
    # -------------------------------
    try:
        import streamlit as st
        from openai import OpenAI

        # Vérifie si clé existe
        if "OPENAI_API_KEY" not in st.secrets:
            raise ValueError("Pas de clé API")

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        prompt = f"""
        Tu es un expert en analyse de risque crédit.

        Données client :
        {data}

        Importance variables (SHAP) :
        {shap_values}

        Donne :
        - Score de risque (faible, moyen, élevé)
        - Explication détaillée
        - Points de vigilance
        """

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    # -------------------------------
    # MODE TEST (SANS API)
    # -------------------------------
    except Exception as e:

        # -------------------------------
        # 🔹 Simulation intelligente
        # -------------------------------
        revenu = data.get("revenu", random.randint(200, 2000))
        credit = data.get("credit", random.randint(0, 1000))
        historique = data.get("historique", random.choice(["bon", "moyen", "mauvais"]))

        score = 0

        # Logique simple
        if revenu > 1000:
            score += 2
        else:
            score -= 1

        if credit < 500:
            score += 2
        else:
            score -= 2

        if historique == "bon":
            score += 2
        elif historique == "moyen":
            score += 0
        else:
            score -= 2

        # Classification
        if score >= 3:
            niveau = " FAIBLE RISQUE"
        elif score >= 0:
            niveau = " RISQUE MODÉRÉ"
        else:
            niveau = " RISQUE ÉLEVÉ"

        # Explication simulée
        explication = f"""
         MODE TEST (sans API OpenAI)

         Résultat : {niveau}

         Analyse :
        - Revenu mensuel : {revenu}
        - Niveau de crédit : {credit}
        - Historique : {historique}

         Interprétation :
        - Un revenu {'élevé' if revenu > 1000 else 'faible'} influence {'positivement' if revenu > 1000 else 'négativement'} le score
        - Un niveau de crédit {'faible' if credit < 500 else 'élevé'} est {'favorable' if credit < 500 else 'risqué'}
        - L'historique est '{historique}'

         Conclusion :
        Cette analyse est simulée pour test académique.
        """

        return explication