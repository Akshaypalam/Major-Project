
from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

# ThingSpeak configuration
CHANNEL_ID = "3470943"
READ_API_KEY = "53TE7P65F8X4FGUJ"

# Provisional raw-reading threshold, retained from your ESP32 code
MQ135_THRESHOLD = 600


def get_sensor_data():
    url = (
        f"https://api.thingspeak.com/channels/"
        f"{CHANNEL_ID}/feeds/last.json"
    )

    response = requests.get(
        url,
        params={"api_key": READ_API_KEY},
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    # Check that all required ThingSpeak fields are available
    if not all(data.get(field) not in (None, "")
               for field in ("field1", "field2", "field3")):
        raise ValueError("Sensor data is missing from ThingSpeak.")

    temperature = float(data["field1"])
    humidity = float(data["field2"])
    mq135 = float(data["field3"])

    # This is a raw sensor reading, NOT ppm or official AQI.
    if mq135 < MQ135_THRESHOLD:
        mq135_status = "Lower sensor reading"
    else:
        mq135_status = "Higher sensor reading"

    return {
        "temperature": temperature,
        "humidity": humidity,
        "mq135": mq135,
        "mq135_status": mq135_status,
        "mq135_threshold": MQ135_THRESHOLD
    }


@app.route("/")
def home():
    try:
        data = get_sensor_data()

        return render_template(
            "index.html",
            temperature=data["temperature"],
            humidity=data["humidity"],
            mq135=data["mq135"],
            mq135_status=data["mq135_status"]
        )

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return render_template(
            "index.html",
            temperature="--",
            humidity="--",
            mq135="--",
            mq135_status="Waiting for sensor data"
        )


@app.route("/data")
def data():
    try:
        sensor_data = get_sensor_data()
        return jsonify(sensor_data)

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return jsonify({
            "error": "Unable to retrieve valid sensor data from ThingSpeak."
        }), 503


if __name__ == "__main__":
    app.run(debug=True)