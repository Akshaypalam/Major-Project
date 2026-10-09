
import requests

CHANNEL_ID = "3470943"
READ_API_KEY = "53TE7P65F8X4FGUJ"


url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json"

response = requests.get(
    url,
    params={"api_key": READ_API_KEY}
)

if response.status_code == 200:

    data = response.json()

    temperature = float(data["field1"])
    humidity = float(data["field2"])
    mq135 = float(data["field3"])

    print("\n===== LIVE AIR QUALITY DATA =====")
    print(f"Temperature : {temperature} °C")
    print(f"Humidity    : {humidity} %")
    print(f"MQ135       : {mq135}")
    print("=================================")

else:
    print("Failed to get data from ThingSpeak")
    print("Status code:", response.status_code)