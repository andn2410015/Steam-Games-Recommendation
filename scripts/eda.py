import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CREATE PLOTS FOLDER
os.makedirs(
    "plots",
    exist_ok=True
)

# STYLE
sns.set_style("whitegrid")

def run_eda():

    # LOAD DATASET
    df = pd.read_csv("data/steam.csv")

    print(df.shape)

    print(df.head())

    # =========================================
    # BASIC INFORMATION
    # =========================================

    print(df.info())

    # =========================================
    # DESCRIPTIVE STATISTICS
    # =========================================

    print(df.describe())

    # =========================================
    # CHECK MISSING VALUES
    # =========================================

    print(df.isnull().sum())

    # =========================================
    # PRICE DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        np.log1p(df['price']),
        bins=30
    )

    plt.title("Log Price Distribution")

    plt.xlabel("Log(Price + 1)")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/01_log_price_distribution.png"
    )

    plt.show()

    # =========================================
    # OWNERS DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['owners'],
        bins=30
    )

    plt.title("Owners Distribution")

    plt.xlabel("Owners")

    plt.ylabel("Frequency")

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    plt.savefig(
        "plots/02_owners_distribution.png"
    )

    plt.show()

    # =========================================
    # AVERAGE PLAYTIME DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['average_playtime'],
        bins=30
    )

    plt.title("Average Playtime Distribution")

    plt.xlabel("Average Playtime")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/03_average_playtime_distribution.png"
    )

    plt.show()

    # =========================================
    # POSITIVE RATINGS DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['positive_ratings'],
        bins=30
    )

    plt.title("Positive Ratings Distribution")

    plt.xlabel("Positive Ratings")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/04_positive_ratings_distribution.png"
    )

    plt.show()

    # =========================================
    # TOP 10 MOST COMMON GENRES
    # =========================================

    top_genres = (
        df['genres']
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=top_genres.index,
        y=top_genres.values
    )

    plt.title("Top 10 Genres")

    plt.xticks(rotation=45)

    plt.ylabel("Count")

    plt.savefig(
        "plots/05_top_10_genres.png"
    )

    plt.show()

    # =========================================
    # TOP 10 MOST COMMON TAGS
    # =========================================

    top_tags = (
        df['steamspy_tags']
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(12,5))

    sns.barplot(
        x=top_tags.index,
        y=top_tags.values
    )

    plt.title("Top 10 Steam Tags")

    plt.xticks(rotation=60)

    plt.ylabel("Count")

    plt.savefig(
        "plots/06_top_10_tags.png"
    )

    plt.show()

    # =========================================
    # CREATE NEW FEATURES
    # =========================================

    df['rating_ratio'] = (
        df['positive_ratings']
        /
        (
            df['positive_ratings']
            +
            df['negative_ratings']
            +
            1
        )
    )

    # =========================================
    # RATING RATIO DISTRIBUTION
    # =========================================

    plt.figure(figsize=(8,5))

    plt.hist(
        df['rating_ratio'],
        bins=30
    )

    plt.title("Rating Ratio Distribution")

    plt.xlabel("Rating Ratio")

    plt.ylabel("Frequency")

    plt.savefig(
        "plots/07_rating_ratio_distribution.png"
    )

    plt.show()

    # =========================================
    # CORRELATION HEATMAP
    # =========================================

    corr = df[
        [
            'price',
            'owners',
            'average_playtime',
            'positive_ratings',
            'negative_ratings',
            'rating_ratio'
        ]
    ].select_dtypes(include='number').corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        "plots/08_correlation_heatmap.png"
    )

    plt.show()

    # =========================================
    # SCATTERPLOT: OWNERS VS PLAYTIME
    # =========================================

    plt.figure(figsize=(8,5))

    plt.scatter(
        df['owners'],
        df['average_playtime'],
        alpha=0.5
    )

    plt.title("Owners vs Average Playtime")

    plt.xlabel("Owners")

    plt.ylabel("Average Playtime")

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    plt.savefig(
        "plots/09_owners_vs_playtime.png"
    )

    plt.show()

    # =========================================
    # SCATTERPLOT: PRICE VS RATING
    # =========================================

    plt.figure(figsize=(8,5))

    plt.scatter(
        df['price'],
        df['rating_ratio'],
        alpha=0.5
    )

    plt.title("Price vs Rating Ratio")

    plt.xlabel("Price")

    plt.ylabel("Rating Ratio")

    plt.savefig(
        "plots/10_price_vs_rating.png"
    )

    plt.show()

    # =========================================
    # BOXPLOT OF GAME PRICES
    # =========================================

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x=df['price']
    )

    plt.title("Boxplot of Game Prices")

    plt.savefig(
        "plots/11_boxplot_game_prices.png"
    )

    plt.show()

    # =========================================
    # FINAL CLEAN DATASET PREVIEW
    # =========================================

    print(df.head())

    return df