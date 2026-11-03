
from pathlib import Path
import joblib
import pandas as pd
import logging
import sys
import time
import uuid

from flask import Flask, jsonify, request, g, Response
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded"
)

MODEL_LOADED.set(0)


@app.before_request
def start_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    duration = time.perf_counter() - g.start_time

    response.headers["X-Request-ID"] = g.request_id

    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration)

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration * 1000
    )

    return response


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found")

        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )

        return model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed")
        raise


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("event=validation_error reason=missing_json")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    logger.info("event=application_starting")
    app.run(host="0.0.0.0", port=5000)
