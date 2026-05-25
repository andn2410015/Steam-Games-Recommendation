import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def run_classification(df):

    # =========================================
    # CREATE BINARY LABEL
    # =========================================

    df['good_game'] = (
        df['rating_ratio'] >= 0.75
    ).astype(int)

    print(
        df['good_game'].value_counts()
    )

    # =========================================
    # CLASS DISTRIBUTION
    # =========================================

    plt.figure(figsize=(6,4))

    sns.countplot(
        x='good_game',
        data=df
    )

    plt.title(
        "Good Game Distribution"
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Count"
    )

    plt.savefig(
        "plots/20_good_game_distribution.png"
    )

    plt.show()

    # =========================================
    # CLASSIFICATION FEATURES
    # =========================================

    classification_features = [
        'price',
        'log_owners',
        'log_playtime',
        'engagement_score'
    ]

    X = df[classification_features]

    y = df['good_game']

    # =========================================
    # TRAIN TEST SPLIT
    # =========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================================
    # RANDOM FOREST CLASSIFIER
    # =========================================

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(
        X_train,
        y_train
    )

    # =========================================
    # PREDICTIONS
    # =========================================

    y_pred = rf_model.predict(
        X_test
    )

    # =========================================
    # CLASSIFICATION EVALUATION
    # =========================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # =========================================
    # CONFUSION MATRIX
    # =========================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.savefig(
        "plots/21_confusion_matrix.png"
    )

    plt.show()

    # =========================================
    # FEATURE IMPORTANCE
    # =========================================

    importance_df = pd.DataFrame({
        'Feature': classification_features,
        'Importance': rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    print(importance_df)

    # =========================================
    # FEATURE IMPORTANCE PLOT
    # =========================================

    plt.figure(figsize=(8,5))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df
    )

    plt.title(
        "Feature Importance"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.ylabel(
        "Feature"
    )

    plt.savefig(
        "plots/22_feature_importance.png"
    )

    plt.show()