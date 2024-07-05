import requests
import json

## Sample Client used with wrapped LLM
if __name__ == "__main__":

    ## Basic setup for the client - server communication
    BASE_URL = "http://localhost:5000"
    REQUEST_URL = f"{BASE_URL}/request"
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",

    }

    # Sample prompt for the LLM    
    prompt_math = "What is 14+5?"

    # Data to be sent to the server
    data = {
        "prompt": [{"role":"user", "content":prompt_math}],
        "process_logits": False
    }

    print("Sent")
    
    # Look Ma no Transformers!
    response = requests.post(REQUEST_URL, data=json.dumps(data), headers=headers).json()

    print(response.get("response"))