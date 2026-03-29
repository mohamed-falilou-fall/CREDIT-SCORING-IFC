# agents/risk_agent.py

def risk_analysis(data, shap_values=None):

    score = 0

    revenu = data.get("revenu", 0)
    credit = data.get("credit", 0)

    if revenu > 1000:
        score += 2
    else:
        score -= 1

    if credit < 500:
        score += 2
    else:
        score -= 2

    if score >= 3:
        niveau = "FAIBLE RISQUE"
    elif score >= 0:
        niveau = "RISQUE MODÉRÉ"
    else:
        niveau = "RISQUE ÉLEVÉ"

    # Ajout SHAP réel
    shap_info = ""
    if shap_values is not None:
        shap_info = f"\nTop variables SHAP : {shap_values[:3]}"

    return f"""
    Score : {score}
    Niveau : {niveau}

    Revenu : {revenu}
    Crédit : {credit}
    {shap_info}
    """
