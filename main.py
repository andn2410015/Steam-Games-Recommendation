from scripts.eda import run_eda

from scripts.feature_engineering import (
    run_feature_engineering
)

from scripts.clustering import (
    run_clustering
)

from scripts.recommendation import (
    run_recommendation
)

from scripts.classification import (
    run_classification
)

# =========================================
# MAIN PIPELINE
# =========================================

def main():

    print("=" * 50)
    print("STEAM GAME ANALYSIS PROJECT")
    print("=" * 50)

    # =====================================
    # EDA
    # =====================================

    print("\nRunning EDA...\n")

    df = run_eda()

    # =====================================
    # FEATURE ENGINEERING
    # =====================================

    print("\nRunning Feature Engineering...\n")

    df, X_pca, features = (
        run_feature_engineering(df)
    )

    # =====================================
    # CLUSTERING
    # =====================================

    print("\nRunning Clustering...\n")

    df = run_clustering(
        df,
        X_pca,
        features
    )

    # =====================================
    # RECOMMENDATION SYSTEM
    # =====================================

    print("\nRunning Recommendation System...\n")

    run_recommendation(df)

    # =====================================
    # CLASSIFICATION
    # =====================================

    print("\nRunning Classification...\n")

    run_classification(df)

    # =====================================
    # EXPORT FINAL DATASET
    # =====================================

    df.to_csv(
        "steam_games_clustered.csv",
        index=False
    )

    print("\nProject Finished Successfully!")

# =========================================
# RUN MAIN
# =========================================

if __name__ == "__main__":

    main()