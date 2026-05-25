import pandas as pd

from scripts.recommendation import (
    recommend_games
)

# SHOW FULL OUTPUT
pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)

# LOAD DATA
df = pd.read_csv(
    "steam_games_clustered.csv"
)

# GET RECOMMENDATIONS
recommendations = recommend_games(
    df,
    user_age=17,
    user_tags=['open world'],
    max_price=20,
    platform='windows'
)

# PRINT RESULTS
print(recommendations)