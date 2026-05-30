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
    # SILHOUETTE ANALYSIS
    # =========================================

    silhouette_scores = []

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_pca
        )

        score = silhouette_score(
            X_pca,
            labels
        )

        silhouette_scores.append(score)

        print(
            f"K={k}: {score:.4f}"
        )

    plt.figure(figsize=(8,5))

    plt.plot(
        range(2,11),
        silhouette_scores,
        marker='o'
    )

    plt.title(
        "Silhouette Scores"
    )

    plt.xlabel(
        "Number of Clusters"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.savefig(
        "plots/18_silhouette_scores.png"
    )

    plt.show()

    # =========================================
    # ELBOW METHOD
    # =========================================

    inertia_values = []

    k_values = range(1, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
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
        "plots/19_elbow_method.png"
    )

    plt.show()

    # =========================================
    # FINAL K VALUE
    # =========================================

    best_k = 4

    # =========================================
    # KMEANS CLUSTERING
    # =========================================

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    df['cluster'] = kmeans.fit_predict(
        X_pca
    )

    print(
        df['cluster'].value_counts()
    )

    cluster_sizes = (
        df['cluster']
        .value_counts()
        .sort_index()
    )

    print("\nCluster Sizes")

    print(cluster_sizes)

    # =========================================
    # CLUSTER VISUALIZATION
    # =========================================

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
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

    plt.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    plt.savefig(
        "plots/20_kmeans_clusters.png"
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

    # =========================================
    # CLUSTER ANALYSIS
    # =========================================

    cluster_analysis = (
        df.groupby('cluster')[features]
        .mean()
        .round(3)
    )

    print("\nCluster Analysis")

    print(cluster_analysis.round(3))

    # =========================================
    # CLUSTER PROFILES
    # =========================================

    print("\nCluster Profiles")

    for cluster_id in cluster_analysis.index:

        print("\n" + "=" * 50)
        print(f"Cluster {cluster_id}")
        print("=" * 50)

        print(
            cluster_analysis
            .round(3)
            .loc[cluster_id]
        )

    cluster_analysis.round(3).to_csv(
        "plots/22_cluster_analysis.csv"
    )

    return df