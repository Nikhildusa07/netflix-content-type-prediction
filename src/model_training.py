import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_model(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(df["content_features"])
    y = df["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model, vectorizer, X_test, y_test


def save_model(model, vectorizer):
    joblib.dump(model, "models/type_prediction_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")