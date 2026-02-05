from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from aspect_miner import AspectOpinionPipeline

app = Flask(__name__)
pipeline = AspectOpinionPipeline()

EXAMPLE_REVIEW = (
    "The battery life is great but the screen quality is not good. "
    "I love the camera, yet the speakers are very weak and disappointing."
)


@app.get("/")
def home():
    return render_template("index.html", example_review=EXAMPLE_REVIEW)


@app.post("/analyze")
def analyze():
    review = request.form.get("review", "").strip()
    if not review:
        return render_template(
            "index.html",
            error="Please enter a review.",
            example_review=EXAMPLE_REVIEW,
            review=review,
        )

    result = pipeline.analyze(review)
    return render_template("index.html", result=result, review=review, example_review=EXAMPLE_REVIEW)


@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(force=True, silent=True) or {}
    review = str(payload.get("review", "")).strip()
    if not review:
        return jsonify({"error": "review field is required"}), 400
    return jsonify(pipeline.analyze(review))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
