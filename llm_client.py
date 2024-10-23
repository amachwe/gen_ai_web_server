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

    def send_request(self, prompt:list[dict]):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":False, "run_config": {"do_sample": False,
    "debug_mode": False}}), headers=self.headers).json()
        return response
    
    def send_request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={}):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": run_config}), headers=self.headers).json()
        return response
    
    def send_request_with_debug(self, prompt:list[dict], process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": False,
    "debug_mode": True}}), headers=self.headers).json()
        return response
    
    def send_request_with_sampling(self, prompt:list[dict], process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": True,
    "debug_mode": False}}), headers=self.headers).json()
        return response
    
    def send_request_with_sampling_and_debug(self, prompt:list[dict], process_logits:bool=False):
        """
        Helper method to send a request to the LLM
        """
        response = requests.post(self.request_url, data=json.dumps({"prompt":prompt, "process_logits":process_logits, "run_config": {"do_sample": True,
    "debug_mode": True}}), headers=self.headers).json()
        return response
    
    def extract_response(self, response):
        """
        Helper method to extract the response from the LLM
        """
        return response["response"][1]["content"]
    
import openai

class OpenAIClient(object):

    def __init__(self):
        self.client = openai.OpenAI()
           

    def send_request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={}):
        """
        Helper method to send a request to the OpenAI LLM
        """
        model_name = run_config.get("model_name", "gpt-4o-mini")
        return self.client.chat.completions.create(model=model_name, messages=prompt, logprobs=process_logits).choices[0].message
    
    def extract_response(self, response):
        """
        Helper method to extract the response from the LLM
        """
        return response.content
    
import google.generativeai as genai
import os


class GeminiClient(object):

    def __init__(self):
        genai.configure(api_key=os.environ["API_KEY"])
        

           

    def send_request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={}):
        """
        Helper method to send a request to the OpenAI LLM
        """
        model_name = run_config.get("model_name", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(model_name)
        return self.model.generate_content(prompt[0]["content"])

    
    def extract_response(self, response):
        """
        Helper method to extract the response from the LLM
        """
        return response.text
