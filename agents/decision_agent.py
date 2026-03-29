# agents/decision_agent.py

def final_decision(risk, financial, strategy):

    if "FAIBLE RISQUE" in risk and "FAIBLE" in financial:
        decision = "APPROUVÉ"
    elif "ÉLEVÉ" in financial:
        decision = "REFUSÉ"
    else:
        decision = "À ANALYSER"

    return f"""
    Décision finale : {decision}

    Basée sur :
    - Risque
    - Analyse financière
    - Stratégie
    """
