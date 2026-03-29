# agents/chat_agent.py

def chat_ifc(context, question):

    question = question.lower()

    if "moyenne" in question:
        return "Analyse : consultez les statistiques du dataset affichées."

    elif "modèle" in question:
        return "Le meilleur modèle est celui avec le MSE le plus faible."

    else:
        return "Question non reconnue. Essayez : moyenne, modèle, risque."
