import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def run_feature_engineering(df):

    # =========================================
    # PRINT PREVIEW RATIO DISTRIBUTION
    # =========================================

    # rating_ratio was created during EDA

    print(
        df[
            [
                'positive_ratings',
                'negative_ratings',
                'rating_ratio'
            ]
        ].head()
    )

    # =========================================
    # RATING RATIO DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['rating_ratio'],
        bins=30
    )

    plt.title(
        "Rating Ratio Distribution"
    )

    plt.xlabel(
        "Rating Ratio"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.savefig(
        "plots/12_rating_ratio_distribution.png"
    )

    plt.show()

    # =========================================
    # CLEAN OWNERS COLUMN
    # =========================================

    # Convert owner ranges into average values

    df['owners'] = (
        df['owners']
        .str.split('-')
    )

    df['owners'] = df['owners'].apply(
        lambda x:
        (
            int(x[0]) + int(x[1])
        ) / 2
    )

    print(df['owners'].head())

    # =========================================
    # LOG TRANSFORMATION
    # =========================================

    df['log_owners'] = np.log1p(
        df['owners']
    )

    df['log_playtime'] = np.log1p(
        df['average_playtime']
    )

    print(
        df[
            [
                'owners',
                'log_owners',
                'average_playtime',
                'log_playtime'
            ]
        ].head()
    )

    # =========================================
    # OWNERS BEFORE LOG
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['owners'],
        bins=30
    )

    plt.title(
        "Owners Before Log Transform"
    )

    plt.xlabel("Owners")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/13_owners_before_log_transform.png"
    )

    plt.show()

    # =========================================
    # OWNERS AFTER LOG
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['log_owners'],
        bins=30
    )

    plt.title(
        "Owners After Log Transform"
    )

    plt.xlabel("Log Owners")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/14_owners_after_log_transform.png"
    )

    plt.show()

    # =========================================
    # PLAYTIME BEFORE LOG
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['average_playtime'],
        bins=30
    )

    plt.title(
        "Playtime Before Log Transform"
    )

    plt.xlabel("Average Playtime")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/15_playtime_before_log.png"
    )

    plt.show()

    # =========================================
    # PLAYTIME AFTER LOG
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['log_playtime'],
        bins=30
    )

    plt.title(
        "Playtime After Log Transform"
    )

    plt.xlabel("Log Playtime")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/16_playtime_after_log.png"
    )

    plt.show()

    # =========================================
    # ENGAGEMENT SCORE
    # =========================================

    df['engagement_score'] = (
        df['average_playtime']
        /
        (
            df['owners'] + 1
        )
    )

    print(
        df[
            [
                'average_playtime',
                'owners',
                'engagement_score'
            ]
        ].head()
    )

    # =========================================
    # FEATURE SELECTION
    # =========================================

    features = [
        'price',
        'rating_ratio',
        'log_owners',
        'log_playtime',
        'engagement_score'
    ]

    X = df[features]

    print(X.head())

    # =========================================
    # NORMALIZATION
    # =========================================

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=features
    )

    df['scaled_rating_ratio'] = scaled_df['rating_ratio']

    df['scaled_log_owners'] = scaled_df['log_owners']

    df['scaled_log_playtime'] = scaled_df['log_playtime']

    print(X_scaled[:5])

    # =========================================
    # PCA
    # =========================================

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    print(
        "Explained Variance Ratio:",
        pca.explained_variance_ratio_
    )

    print(
        "Total Explained Variance:",
        pca.explained_variance_ratio_.sum()
    )

    # =========================================
    # ADD PCA COLUMNS
    # =========================================

    df['pca1'] = X_pca[:, 0]

    df['pca2'] = X_pca[:, 1]

    print(
        df[
            [
                'pca1',
                'pca2'
            ]
        ].head()
    )

    # =========================================
    # PCA VISUALIZATION
    # =========================================

    plt.figure(figsize=(8,6))

    plt.scatter(
        df['pca1'],
        df['pca2'],
        alpha=0.3
    )

    plt.title(
        "PCA Projection"
    )

    plt.xlabel(
        "PCA 1"
    )

    plt.ylabel(
        "PCA 2"
    )

    plt.savefig(
        "plots/17_pca_projection.png"
    )

    plt.show()

    return df, X_pca, features