# -*- coding: utf-8 -*-
"""
Chat Agent IFC - version LLM + RAG
"""

from llm.rag_engine import load_vectorstore, load_llm, ask_ifc_llm

# Chargement global (cache)
db = None
llm = None


def init_llm():

    global db, llm

    if db is None:
        db = load_vectorstore()

    if llm is None:
        llm = load_llm()


def chat_ifc(context, question):

    init_llm()

    # ==========================
    # 1. RAG IFC
    # ==========================
    try:
        answer = ask_ifc_llm(question, db, llm)

        return f"""
        🤖 Réponse IFC (LLM + Reports):

        {answer}
        """

    except Exception as e:

        # ==========================
        # 2. FALLBACK (ancien système)
        # ==========================
        question_lower = question.lower()

        if "moyenne" in question_lower:
            return "Analyse : consultez les statistiques du dataset affichées."

        elif "modèle" in question_lower:
            return "Le meilleur modèle est celui avec le MSE le plus faible."

        else:
            return f"Erreur LLM → fallback activé : {str(e)}"