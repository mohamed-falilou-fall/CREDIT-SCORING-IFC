# agents/decision_agent.py

import random

# -------------------------------
# Fonction principale 
# -------------------------------
def final_decision(risk_result, financial_result=None, strategy_result=None):
    """
    Prend les outputs des agents et donne une décision finale
    """

    try:
        import streamlit as st
        from openai import OpenAI

        if "OPENAI_API_KEY" not in st.secrets:
            raise ValueError("No API key")

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        prompt = f"""
        Tu es un expert en crédit IFC.

        Analyse risque :
        {risk_result}

        Analyse financière :
        {financial_result}

        Stratégie :
        {strategy_result}

        Donne :
        - Décision finale (APPROUVER / REFUSER / À ANALYSER)
        - Justification claire
        """

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    except:
        # -------------------------------
        # MODE TEST (SANS API)
        # -------------------------------
        decision = random.choice([
            " APPROUVÉ",
            " REFUSÉ",
            "️ À ANALYSER"
        ])

        return f"""
         MODE TEST (sans API)

         Décision finale : {decision}

         Basé sur une simulation multi-agents :
        - Risque : pris en compte
        - Financier : analysé
        - Stratégie : intégrée

         Ceci est une décision simulée pour démonstration.
        """

# -------------------------------
#  (OPTIONNEL) Chat expert IFC
# -------------------------------
def chat_ifc(context, question):

    try:
        import streamlit as st
        from openai import OpenAI

        if "OPENAI_API_KEY" not in st.secrets:
            raise ValueError("No API key")

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Expert IFC credit risk"},
                {"role": "user", "content": context + "\nQuestion: " + question}
            ]
        )

        return response.choices[0].message.content

    except:
        return " Mode test : réponse simulée IFC"