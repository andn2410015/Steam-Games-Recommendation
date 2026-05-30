import pandas as pd

def recommend_games(
    df,
    user_age,
    user_tags=None,
    max_price=None,
    platform=None,
    top_n=5
):

    # LOWERCASE
    filtered_df = df.copy()

    filtered_df['steamspy_tags'] = (
        filtered_df['steamspy_tags']
        .str.lower()
    )

    filtered_df['platforms'] = (
        filtered_df['platforms']
        .str.lower()
    )

    # AGE FILTER
    filtered_df = filtered_df[
        filtered_df['required_age']
        <= user_age
    ]

    # TAG FILTER
    if user_tags:

        for tag in user_tags:

            # Missing steamspy_tags values are treated as False
            # during tag filtering (excluded from recommendations)

            filtered_df = filtered_df[
                filtered_df['steamspy_tags']
                .str.contains(
                    tag.lower(),
                    na=False
                )
            ]

    # BUDGET FILTER
    if max_price is not None:

        filtered_df = filtered_df[
            filtered_df['price']
            <= max_price
        ]

    # PLATFORM FILTER
    if platform:

        filtered_df = filtered_df[
            filtered_df['platforms']
            .str.contains(
                platform.lower(),
                na=False
            )
        ]

    # EMPTY RESULT CHECK
    if filtered_df.empty:

        print(
            "No matching games found."
        )

        return pd.DataFrame()
    
    filtered_df = filtered_df.copy()

    # RANKING SCORE
    # ranking_score combines:
    # - rating quality
    # - popularity
    # - playtime
    # - engagement
    filtered_df['ranking_score'] = (
        filtered_df['scaled_rating_ratio'] * 0.5
        +
        filtered_df['scaled_log_owners'] * 0.2
        +
        filtered_df['scaled_log_playtime'] * 0.15
        +
        filtered_df['scaled_engagement_score'] * 0.15
    )

    # SORT RECOMMENDATIONS
    filtered_df = filtered_df.sort_values(
        by='ranking_score',
        ascending=False
    )

    return filtered_df[
        [
            'name',
            'price',
            'platforms',
            'steamspy_tags',
            'rating_ratio',
            'owners',
            'average_playtime',
            'cluster',
            'ranking_score'
        ]
    ].head(top_n)


def run_recommendation(df):

    recommendations = recommend_games(
        df,
        user_age=17,
        user_tags=['open world'],
        max_price=20,
        platform='windows'
    )

    print("\nTop Recommended Games: ")
    return recommendations