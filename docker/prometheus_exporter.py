import os
from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5005")

# Metrik untuk API model
REQUEST_COUNT    = Counter(   "http_requests_total",              "Total HTTP Requests")
REQUEST_LATENCY  = Histogram( "http_request_duration_seconds",    "HTTP Request Latency")
THROUGHPUT       = Counter(   "http_requests_throughput",         "Total number of requests per second")

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")


# Endpoint untuk Prometheus
@app.route("/metrics", methods=["GET"])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# Endpoint untuk mengakses API model dan mencatat metrik
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    api_url = f"{MLFLOW_URL}/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data, timeout=30)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
