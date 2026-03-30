IFC AI Credit Scoring SaaS Platform (α)

- Key South Lab – Team Epsilon
- Auteur : Mohamed Falilou Fall

Description :
Plateforme avancée d’analyse et de scoring de crédit basée sur l’IA, alignée avec les standards de l’International Finance Corporation (IFC). Déployable sur Streamlit avec diagnostic SHAP avancé, recommandations stratégiques et chat RAG IFC.

---

Fonctionnalités clés

- Upload & préprocessing CSV automatisé
- Benchmark multi-modèles ML : RandomForest, XGBoost, LightGBM, CatBoost
- Explainable AI (SHAP) : beeswarm + analyse détaillée par client
- Diagnostic stratégique SHAP-driven : score actuel vs potentiel, top 3 actions, segmentation High Risk / Turnaround / Strong
- Expected Loss (PD × LGD × EAD)
- Epsilon-Agent AI System : Risk / Financial / Strategy / Decision Agents
- Filtrage clients avancé : pays, secteur, id_client
- IFC AI Chat + RAG : interroger données, modèles et rapports PDF

---

Architecture

```
Streamlit UI
   │
   ├─ Preprocessing
   ├─ ML Models (RF/XGB/LGBM/CatBoost)
   ├─ SHAP Engine
   │
   └─ Epsilon Agents (Risk / Financial / Strategy / Decision)
   │
   └─ RAG IFC (LLM + Vectorstore)
   │
Final Decision & Dashboard
```

---

Structure du projet

```
ifc_ai_credit_app/
│── app.py
├── utils/preprocessing.py
├── agents/
│   ├── risk_agent.py
│   ├── financial_agent.py
│   ├── strategy_agent.py
│   ├── decision_agent.py
│   └── chat_agent.py
├── llm/rag_engine.py
├── report/      # PDF IFC pour RAG
├── requirements.txt
└── README.md
```

---

Utilisation rapide

1. Configuration sidebar : modèle ML principal, activer Multi Epsilon-Agent IA
2. Upload CSV : aperçu des données + colonnes `probabilite_defaut`, `perte_en_cas_defaut`, `exposition_defaut`
3. Filtrage clients : pays, secteur, id_client → toutes analyses sur les données filtrées
4. Préprocessing : nettoyage et génération X/y
5. Modélisation & Benchmark : validation croisée, MSE, sélection automatique du meilleur modèle
6. SHAP explicabilité : beeswarm + analyse client
7. Diagnostic stratégique : top 3 actions, score potentiel, segmentation
8. Expected Loss : calcul automatique PD × LGD × EAD
9. Epsilon-Agent AI System : multi-agents pour scoring et recommandations
10. IFC AI Chat + RAG : poser des questions sur dataset, modèles et rapports PDF

---

Outputs principaux

- Résumé exécutif par client : score actuel, score potentiel, gain potentiel, top 3 actions
- Analyse détaillée : features critiques, impact SHAP, benchmark vs moyenne, gains potentiels
- Recommandation finale : Approuvé / Risqué / Refusé

---

Limitations (α)

- Itérations ML réduites pour tests rapides
- SHAP dépend du modèle
- RAG nécessite le dossier `report/`
- Préproduction uniquement → augmenter itérations et hyperparamètres pour usage réel

---

Roadmap

- Optimisation hyperparamètres ML
- Déploiement cloud + API
- Scoring temps réel
- Dashboard exécutif interactif
- Intégration données macro / ESG

---

Auteur

Mohamed Falilou Fall – Key South Lab, Team Epsilon
