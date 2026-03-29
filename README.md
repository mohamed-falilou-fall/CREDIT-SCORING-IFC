IFC AI Credit Scoring SaaS Platform (Version α)

- Key South Lab – Team Epsilon
- Auteur : Mohamed Falilou Fall
- Description : Plateforme d’analyse et de scoring de crédit basée sur l’IA, alignée avec les standards IFC, entièrement déployable sur Streamlit.

---

Présentation

La plateforme IFC AI Credit SaaS permet :

- L’upload d’un dataset de clients (CSV) pour prédire la probabilité de défaut et la perte attendue.
- Le benchmark de plusieurs modèles de machine learning (RandomForest, XGBoost, LightGBM, CatBoost).
- L’explicabilité des décisions via SHAP.
- Une analyse multi-agents IA (« Epsilon-Agent AI System ») pour fournir un scoring, un profil financier, une stratégie et une recommandation finale.
- Une interface de chat AI pour poser des questions sur les données et les modèles.
- Filtrage avancé des clients : possibilité de sélectionner un pays, un secteur d’activité ou de rechercher un client par son `id_client`.

Cette version α a des itérations réduites pour des tests rapides sur Streamlit.

---

Structure du projet


ifc_ai_credit_app/
│
├─ app.py
├─ utils/
│   └─ preprocessing.py
├─ agents/
│   ├─ risk_agent.py
│   ├─ financial_agent.py
│   ├─ strategy_agent.py
│   ├─ decision_agent.py
│   └─ chat_agent.py
├─ requirements.txt
└─ README.md



---

Utilisation pas à pas

1. Configuration

- Dans la sidebar, sélectionnez le modèle ML principal :
  Auto (Best), RandomForest, XGBoost, LightGBM, CatBoost
- Activez ou non le système Multi Epsilon-Agent IA.

2. Upload du dataset

- Cliquez sur Charger un fichier CSV.
- Le dataset doit contenir les colonnes pour la prédiction (utiliser le fichier `dataset_ifc_simulé.csv`) :

  - `probabilite_defaut`, `perte_en_cas_defaut`, `exposition_defaut` (pour Expected Loss)
- Aperçu des premières lignes s’affichera automatiquement.

3. Filtrage des clients

- Dans la sidebar, vous pouvez maintenant :

  - Sélectionner un pays pour filtrer les clients
  - Sélectionner un secteur d’activité
  - Rechercher un id_client spécifique

- Les données affichées et le traitement seront appliqués uniquement sur les clients filtrés.

4. Préprocessing

- Les données sont nettoyées et préparées pour la modélisation.
- Une notification confirme le nombre de lignes et de variables prêtes pour l’entraînement.

5. Modélisation

- La plateforme entraîne tous les modèles choisis.
- Le benchmark MSE s’affiche sous forme de tableau.
- Le meilleur modèle est automatiquement sélectionné si vous avez choisi Auto (Best).
- Le modèle final est entraîné sur l’intégralité du dataset filtré.

6. Explicabilité SHAP

- Les valeurs SHAP sont calculées pour visualiser l’impact de chaque variable sur les prédictions.
- Un graphique beeswarm est affiché pour interpréter l’importance des features.

7. Expected Loss

- Si les colonnes nécessaires sont présentes, la plateforme calcule la perte attendue pour chaque client :

```

expected_loss = probabilite_defaut  perte_en_cas_defaut  exposition_defaut

```

- Les premières valeurs sont affichées.

8. Epsilon-Agent AI System

- Sélectionnez un client via le slider dans le dataset filtré.
- Cliquez sur Lancer analyse IA :

  - Risk Agent : Profil de risque du client
  - Financial Agent : Analyse financière (Expected Loss)
  - Strategy Agent : Recommandations stratégiques
  - Decision Agent : Recommandation finale (Approuvé / Risqué / Refusé)

9. IFC AI Chat

- Posez des questions sur les données ou le modèle dans le champ text_input.
- L’agent IA fournira des réponses basées sur le dataset filtré et les résultats du benchmark.

---

Fonctionnalités principales

- Upload CSV et préprocessing automatisé
- Benchmark multi-modèles (ML classique)
- Explicabilité SHAP interactive
- Calcul d’Expected Loss
- Système multi-agents IA pour scoring et recommandations
- Filtrage avancé par pays, secteur et id_client
- Interface chat AI pour analyse et interprétation des données

---

Agents AI intégrés

1. Risk Agent : analyse le risque client avec fallback automatique.
2. Financial Agent : calcule l’impact financier via Expected Loss.
3. Strategy Agent : propose une stratégie pour le client.
4. Decision Agent : synthétise les résultats et fournit une recommandation finale.
5. Chat Agent : répond aux questions sur le dataset filtré et les performances des modèles.

---

Support et dépannage

- Erreur TypeError : assurez-vous que tous les agents acceptent les arguments `row_data` et `shap_row`.
- Erreur HTTP 404 GPT4All : utilisez un modèle disponible ou laissez le fallback activé.
- Problèmes Streamlit : vérifier la version et l’installation des dépendances.

---

Note : Cette plateforme est une version α pour tests rapides et démonstrations. Pour un usage en production, augmenter le nombre d’itérations et ajuster les hyperparamètres ML.
