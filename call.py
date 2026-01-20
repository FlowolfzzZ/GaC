import requests

if __name__ == "__main__":

    url = "http://0.0.0.0:8000//v1/chat/completions"

    data = {
        "messages": [
            {
                "role": "user",
                "content": "Which word does not belong with the others?\ntyre, steering wheel, car, engine",
            }
        ],
        "max_tokens": 16,
        "apply_chat_template": True,
    }

    completion = requests.post(url, json=data).json()
    print(completion["choices"][0])