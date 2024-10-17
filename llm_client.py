import requests
import json

   
class Client(object):
    def __init__(self, target_url:str="http://localhost:5000"):
        self.request_url = f"{target_url}/request"
        self.vocab_url = f"{target_url}/vocab"
        self.headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",

        }

    def send_request(self, prompt:str):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":False, "run_config": {"do_sample": False,
    "debug_mode": False}}), headers=self.headers).json()
        return response
    
    def send_request(self, prompt:str, process_logits:bool=False, run_config:dict={}):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": run_config}), headers=self.headers).json()
        return response
    
    def send_request_with_debug(self, prompt:str, process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": False,
    "debug_mode": True}}), headers=self.headers).json()
        return response
    
    def send_request_with_sampling(self, prompt:str, process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": True,
    "debug_mode": False}}), headers=self.headers).json()
        return response
    
    def send_request_with_sampling_and_debug(self, prompt:str, process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": True,
    "debug_mode": True}}), headers=self.headers).json()
        return response