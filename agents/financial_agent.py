# agents/financial_agent.py

def financial_analysis(expected_loss):

    if expected_loss < 10000:
        niveau = "FAIBLE"
        reco = "Financement recommandé"
    elif expected_loss < 50000:
        niveau = "MODÉRÉ"
        reco = "Surveillance nécessaire"
    else:
        niveau = "ÉLEVÉ"
        reco = "Risque critique"

    return f"""
    Expected Loss : {expected_loss}

    Niveau : {niveau}
    Analyse : Basée sur seuils internes IFC
    Recommandation : {reco}
    """
