import requests

response = requests.post(
    "http://localhost:8000/scan",
    json={"prompt": "Hello", "model_name": "distilgpt2"}
)
print(response.status_code)
print(response.text)