# agents/strategy_agent.py

from sklearn.cluster import KMeans
import numpy as np

# modèle global (simple)
kmeans_model = None

def train_strategy_model(X):
    global kmeans_model
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X)

def strategy_analysis(row_data):

    global kmeans_model

    if kmeans_model is None:
        return "Modèle stratégie non entraîné"

    values = np.array(list(row_data.values())).reshape(1, -1)
    cluster = kmeans_model.predict(values)[0]

    strategies = {
        0: "Client stable → Expansion possible",
        1: "Client risqué → Réduction exposition",
        2: "Client intermédiaire → Optimisation"
    }

    return f"""
    Cluster : {cluster}
    Stratégie : {strategies.get(cluster)}
    """
