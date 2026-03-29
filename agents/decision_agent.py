# -*- coding: utf-8 -*-
"""
decision_agent.py
Décision finale multi-agents
Mode test complet : aucun appel OpenAI
"""

import random

def final_decision(risk_result, financial_result=None, strategy_result=None):
    """
    Prend les outputs des agents et renvoie une décision finale simulée.
    """
    decision = random.choice(["APPROUVÉ", "REFUSÉ", "À ANALYSER"])

    return f"""
     MODE TEST (sans OpenAI)

    Décision finale : {decision}

    Basé sur une simulation multi-agents :
    - Risque : {risk_result[:30]}...
    - Financier : {financial_result[:30] if financial_result else 'N/A'}...
    - Stratégie : {strategy_result[:30] if strategy_result else 'N/A'}...

    Ceci est une décision simulée pour démonstration académique.
    """

def chat_ifc(context, question):
    """
    Simulation d'un chat expert IFC
    """
    return f" MODE TEST : réponse simulée pour la question '{question}'"
