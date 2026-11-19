"""Deployment TA-06: Prediksi Lagu Hit di Spotify (Random Forest) dengan Flask."""
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(BASE, "model_hit_spotify.joblib"))
model, meta = bundle["model"], bundle["meta"]
THRESHOLD = meta.get("decision_threshold", 0.5)

app = Flask(__name__)

UNIT = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]


@app.get("/")
def index():
    d = meta["defaults"]
    return render_template(
        "index.html",
        genres=meta["genres"], unit=UNIT, d=d, rng=meta["ranges"],
        hit_threshold=meta["hit_threshold"], threshold=THRESHOLD,
        default_genre="pop" if "pop" in meta["genres"] else meta["genres"][0],
        n_trees=model.named_steps["rf"].n_estimators,
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    genre = data.get("genre")
    if genre not in meta["genres"]:
        return jsonify(error="Genre tidak dikenal"), 400
    try:
        row = {f: float(data[f]) for f in meta["audio_features"]}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Input fitur audio tidak lengkap atau bukan angka"), 400
    for f in ("explicit", "key", "mode", "time_signature"):
        row[f] = int(row[f])
    row["track_genre"] = genre
    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[0, 1])
    return jsonify(probability=prob, threshold=THRESHOLD, is_hit=prob >= THRESHOLD)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
