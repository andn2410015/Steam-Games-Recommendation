def recommend_games(
    df,
    user_age,
    user_tags=None,
    max_price=None,
    platform=None,
    top_n=5
):

    # LOWERCASE
    df['steamspy_tags'] = (
        df['steamspy_tags']
        .str.lower()
    )

    df['platforms'] = (
        df['platforms']
        .str.lower()
    )

    filtered_df = df.copy()

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

        return

    # RANKING SCORE
    filtered_df['ranking_score'] = (
        filtered_df['scaled_rating_ratio'] * 0.7
        +
        filtered_df['scaled_log_owners'] * 0.2
        +
        filtered_df['scaled_log_playtime'] * 0.1
    )

    # SORT RECOMMENDATIONS
    filtered_df = filtered_df.sort_values(
        by='ranking_score',
        ascending=False
    )

    return filtered_df[
        [
            'name',
            'steamspy_tags',
            'price',
            'platforms',
            'ranking_score',
            'cluster'
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

    print(recommendations)