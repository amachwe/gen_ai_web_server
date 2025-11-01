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

    def prompt_to_response(self, prompt:str, debug=False)-> str:
        """
        Helper method to convert a prompt to a response
        """
        response = self.send_request([{"role": "user", "content": prompt}])
        return self.extract_response(response)
    
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
        response = self.client.chat.completions.create(model=model_name, messages=prompt, logprobs=process_logits, top_logprobs=5)
        
        return response


    def extract_response(self, response):
        """
        Helper method to extract the response from the LLM
        """
        return response.choices[0].message.content
    
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

import anthropic

class AnthropicClient(object):

    def __init__(self):
        self.client = anthropic.Anthropic()
           

    def send_request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={}):
        """
        Helper method to send a request to the Anthropic LLM
        """
        model_name = run_config.get("model_name", "claude-3-5-haiku-latest")
        max_tokens = run_config.get("max_tokens", 1000)
        temperature = run_config.get("temperature", 0)
        system = run_config.get("system", "")
        return self.client.messages.create(model=model_name, max_tokens=max_tokens, temperature=temperature, system=system, messages=prompt)
    
    def extract_response(self, response):
        """
        Helper method to extract the response from the LLM
        """
        return response.content