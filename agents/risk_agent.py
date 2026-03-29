# -*- coding: utf-8 -*-
"""
risk_agent.py
Analyse du risque crédit IFC
Mode test complet : aucun appel OpenAI
"""

import random

def risk_analysis(data, shap_values=None):
    """
    Analyse du risque de crédit en mode test.
    
    :param data: dict contenant les infos client
    :param shap_values: dict (optionnel) importance variables
    :return: str analyse risque
    """

    # -------------------------------
    # MODE TEST (SANS API)
    # -------------------------------
    revenu = data.get("revenu", random.randint(200, 2000))
    credit = data.get("credit", random.randint(0, 1000))
    historique = data.get("historique", random.choice(["bon", "moyen", "mauvais"]))

    score = 0
    if revenu > 1000: score += 2
    else: score -= 1
    if credit < 500: score += 2
    else: score -= 2
    if historique == "bon": score += 2
    elif historique == "moyen": score += 0
    else: score -= 2

    if score >= 3: niveau = " FAIBLE RISQUE"
    elif score >= 0: niveau = " RISQUE MODÉRÉ"
    else: niveau = " RISQUE ÉLEVÉ"

    explication = f"""
     MODE TEST (sans OpenAI)

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
    Analyse simulée pour tests académiques.
    """
    return explication
