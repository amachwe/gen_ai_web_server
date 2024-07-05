import requests
import json

## Sample Client used with wrapped Phi3 Model from Microsoft
if __name__ == "__main__":
    TARGET_URL = "http://localhost:5000"
    REQUEST_URL = f"{TARGET_URL}/request"
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",

    }

    
    prompt_math = "What is 14+5?"
    data = {
        "prompt": [{"role":"user", "content":prompt_math}],
        "process_logits": False
    }
    print("Sent")
    response = requests.post(REQUEST_URL, data=json.dumps(data), headers=headers).json()
    print(response.get("response"))