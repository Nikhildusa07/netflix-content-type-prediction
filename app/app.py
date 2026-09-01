from flask import Flask, render_template, request
import joblib
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "type_prediction_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    director = request.form.get("director", "Not Given")
    country = request.form.get("country", "Not Given")
    rating = request.form.get("rating", "Not Given")
    genres = request.form.get("listed_in", "Not Given")

    content_features = (
        director + " " +
        country + " " +
        rating + " " +
        genres
    )

    features = vectorizer.transform([content_features])

    prediction = model.predict(features)[0]

    if prediction == 0:
        predicted_type = "Movie"
    else:
        predicted_type = "TV Show"

    return render_template(
        "results.html",
        prediction=predicted_type
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)