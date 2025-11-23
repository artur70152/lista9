import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from minisom import MiniSom

def kmeans(X_scaled, X_pca, ax):

    labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_scaled)

    idx = np.random.choice(len(X_scaled), size=2000, replace=False)
    score = silhouette_score(X_scaled[idx], labels[idx])
    print(f"\n[KMEANS] Silhouette Score: {score:.4f}")

    df_result = pd.DataFrame({"Cluster": labels, "Class": y})
    print(df_result.groupby("Cluster")["Class"].value_counts())

    ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="coolwarm", s=3)
    ax.set_title("K-Means")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")



def dbscan(X_scaled, X_pca, ax):

    labels = DBSCAN(eps=1.8, min_samples=20).fit_predict(X_scaled)

    print("\n[DBSCAN] Clusters encontrados:", np.unique(labels))

    clusters_validos = set(labels)
    clusters_validos.discard(-1)

    if len(clusters_validos) > 1:
        idx = np.random.choice(len(X_scaled), size=2000, replace=False)
        score = silhouette_score(X_scaled[idx], labels[idx])
        print(f"[DBSCAN] Silhouette Score: {score:.4f}")
    else:
        print("[DBSCAN] Não foi possível calcular Silhouette Score.")

    df_result = pd.DataFrame({"Cluster": labels, "Class": y})
    print(df_result.groupby("Cluster")["Class"].value_counts())

    ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=3)
    ax.set_title("DBSCAN")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")



def som(X_scaled, X_pca, ax):
    som_grid_x = 10
    som_grid_y = 10

    som = MiniSom(
        x=som_grid_x, y=som_grid_y,
        input_len=X_scaled.shape[1],
        sigma=1.5, learning_rate=0.5
    )

    som.random_weights_init(X_scaled)
    print("Treinando SOM...")
    som.train_random(X_scaled, 2000)
    print("[SOM] Treinamento concluído.")

    labels_som = []
    for vec in X_scaled:
        i,j = som.winner(vec)
        labels_som.append(i * som_grid_x + j)

    labels_som = np.array(labels_som)
    unicos = np.unique(labels_som)

    print("\n[SOM] Clusters encontrados:", unicos)

    if len(unicos) > 1:
        idx = np.random.choice(len(X_scaled), size=2000, replace=False)
        score = silhouette_score(X_scaled[idx], labels_som[idx])
        print(f"[SOM] Silhouette Score: {score:.4f}")
    else:
        print("[SOM] Não foi possível calcular Silhouette.")

    df_som = pd.DataFrame({"Cluster": labels_som, "Class": y})
    print(df_som.groupby("Cluster")["Class"].value_counts())

    ax.scatter(X_pca[:,0], X_pca[:,1], c=labels_som, cmap="tab20", s=3)
    ax.set_title("SOM")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")




arquivo = r"creditcard.csv"
df = pd.read_csv(arquivo)

print("Linhas antes:", len(df))
df = df.dropna().drop_duplicates()
print("Linhas após:", len(df))


df_sample = df.sample(n=20000, random_state=42)

X = df_sample.drop(columns=["Class"])
y = df_sample["Class"]



X_scaled = StandardScaler().fit_transform(X)



X_pca = PCA(n_components=2).fit_transform(X_scaled)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

kmeans(X_scaled, X_pca, axes[0])
dbscan(X_scaled, X_pca, axes[1])
som(X_scaled, X_pca, axes[2])

plt.tight_layout()
plt.show()
