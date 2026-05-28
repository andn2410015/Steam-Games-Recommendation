import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(
    df,
    X_pca,
    features
):

    # =========================================
    # KMEANS CLUSTERING
    # =========================================

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    df['cluster'] = kmeans.fit_predict(
        X_pca
    )

    print(
        df['cluster'].value_counts()
    )

    # =========================================
    # CLUSTER VISUALIZATION
    # =========================================

    plt.figure(figsize=(8,6))

    plt.scatter(
        df['pca1'],
        df['pca2'],
        c=df['cluster'],
        alpha=0.6
    )

    plt.title(
        "KMeans Clustering of Steam Games"
    )

    plt.xlabel(
        "PCA 1"
    )

    plt.ylabel(
        "PCA 2"
    )

    plt.savefig(
        "plots/18_kmeans_clusters.png"
    )

    plt.show()

    # =========================================
    # CLUSTER ANALYSIS
    # =========================================

    cluster_analysis = df.groupby(
        'cluster'
    )[features].mean()

    print(cluster_analysis)

    cluster_analysis.to_csv(
        "plots/19_cluster_analysis.csv"
    )

    # =========================================
    # SILHOUETTE SCORE
    # =========================================

    score = silhouette_score(
        X_pca,
        df['cluster']
    )

    print(
        "Silhouette Score:",
        score
    )

    # =========================================
    # ELBOW METHOD
    # =========================================

    inertia_values = []

    k_values = range(1, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42
        )

        kmeans.fit(X_pca)

        inertia_values.append(
            kmeans.inertia_
        )

    plt.figure(figsize=(8,5))

    plt.plot(
        k_values,
        inertia_values,
        marker='o'
    )

    plt.title(
        "Elbow Method"
    )

    plt.xlabel(
        "Number of Clusters"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.savefig(
        "plots/20_elbow_method.png"
    )

    plt.show()

    # =========================================
    # CLUSTER DISTRIBUTION
    # =========================================

    plt.figure(figsize=(6,4))

    sns.countplot(
        x='cluster',
        data=df
    )

    plt.title(
        "Cluster Distribution"
    )

    plt.xlabel("Cluster")

    plt.ylabel("Number of Games")

    plt.savefig(
        "plots/21_cluster_distribution.png"
    )

    plt.show()

    return df