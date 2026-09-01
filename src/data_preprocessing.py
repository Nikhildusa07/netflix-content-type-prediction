import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df = df.fillna("Not Given")

    text_columns = df.select_dtypes(include="object").columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    return df