# -*- coding: utf-8 -*-
"""
financial_agent.py
Mode test complet : aucun appel OpenAI
"""

import random

def financial_analysis(expected_loss):
    """
    Analyse financière du client IFC
     Mode test
    """
    niveau = "Faible" if expected_loss < 10000 else "Modéré" if expected_loss < 50000 else "Élevé"
    soutenabilite = "OK" if expected_loss < 20000 else "Attention"
    risque = "" if expected_loss < 10000 else "" if expected_loss < 50000 else ""

    return f"""
     MODE TEST (sans OpenAI)

     Expected Loss : {expected_loss}
     Niveau de perte : {niveau}
     Soutenabilité : {soutenabilite}
     Risque financier : {risque}
    """
