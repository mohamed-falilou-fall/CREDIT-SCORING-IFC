from llm.rag_engine import load_vectorstore, load_llm, ask_ifc_llm

db = None
llm = None

def init_llm():
    global db, llm
    if db is None:
        db = load_vectorstore()
    if llm is None:
        llm = load_llm()


def final_decision(risk, financial, strategy):

    if "FAIBLE RISQUE" in risk and "FAIBLE" in financial:
        decision = "APPROUVÉ"
    elif "ÉLEVÉ" in financial:
        decision = "REFUSÉ"
    else:
        decision = "À ANALYSER"

    # ==============================
    # AJOUT LLM IFC
    # ==============================
    try:
        init_llm()

        question = f"""
        Given this credit decision:

        Risk: {risk}
        Financial: {financial}
        Strategy: {strategy}

        Explain decision using IFC best practices.
        """

        explanation = ask_ifc_llm(question, db, llm)

    except:
        explanation = "Explication LLM indisponible"

    return f"""
    Décision finale : {decision}

    Basée sur :
    - Risque
    - Analyse financière
    - Stratégie

     Justification IFC (LLM):
    {explanation}
    """