import pandas as pd


def prepare_features(df):
    df = df.copy()

    # Target variable
    df["type"] = df["type"].map({
        "Movie": 0,
        "TV Show": 1
    })

    # Combine useful categorical/text features
    df["content_features"] = (
        df["director"].astype(str) + " " +
        df["country"].astype(str) + " " +
        df["rating"].astype(str) + " " +
        df["listed_in"].astype(str)
    )

    return df