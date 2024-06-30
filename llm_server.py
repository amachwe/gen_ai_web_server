import requests
import flask
import json
import abc
import transformers

class LogitStoreProcessor(transformers.LogitsProcessor):
    def __init__(self, ):
        self.logits = []
        self.scores = []

    def __call__(self, input_ids, scores):
        
        self.logits.append(input_ids.tolist())
        self.scores.append(scores.tolist())
        return scores
    
class LLM_Server_Wrapper(abc.ABC):

    def __init__(self, name, tokenizer, model, config:dict):
        self.name = name
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logits_store = LogitStoreProcessor()
        self.logits_processor_list = transformers.LogitsProcessorList([self.logits_store,])
        self.prompting_hint = config.get("prompting_hint", "")

        if config.get("device", "cpu") == "cuda":
            self.model.to("cuda")
        

    def request(self, prompt:str, process_logits:bool=False):

        max_length = self.config.get("max_length", None)
        num_return_sequences = self.config.get("num_return_sequences", 1)
        output_scores = self.config.get("output_scores", None)
        max_new_tokens = self.config.get("max_new_tokens", 500)

        enc = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        kwargs = {}
        if process_logits:
            kwargs["logits_processor"] = self.logits_processor_list
        
        if max_length:
            kwargs["max_length"] = max_length

        if output_scores:
            kwargs["output_scores"] = output_scores

       
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["num_return_sequences"] = num_return_sequences

        
        res = self.model.generate(enc, **kwargs)
       
        dec = self.tokenizer.decode(res[0], skip_special_tokens=True)
        
        return {"response":dec, "logits":self.logits_store.logits, "scores": self.logits_store.scores}
    
    def info(self):
        return {
            "name": self.name,
            "config": self.config,
            "prompting_hint":self.prompting_hint
        }
    

class LLM_Server_Pipe_Wrapper(abc.ABC):

    def __init__(self, name, tokenizer, model, config:dict):
        self.name = name
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logits_store = LogitStoreProcessor()
        self.logits_processor_list = transformers.LogitsProcessorList([self.logits_store,])
        self.prompting_hint = config.get("prompting_hint", "")

        if config.get("device", "cpu") == "cuda":
            print("Moving model to cuda")
            self.model.to("cuda:0")
        

    def request(self, prompt:str, process_logits:bool=False):

        max_length = self.config.get("max_length", None)
        num_return_sequences = self.config.get("num_return_sequences", 1)
        output_scores = self.config.get("output_scores", None)
        max_new_tokens = self.config.get("max_new_tokens", 500)

        

        kwargs = {}
        if process_logits:
            kwargs["logits_processor"] = self.logits_processor_list
        
        if max_length:
            kwargs["max_length"] = max_length

        if output_scores:
            kwargs["output_scores"] = output_scores

       
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["num_return_sequences"] = num_return_sequences

        
        pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.model.device)
        out = pipe(prompt, **kwargs)
        
        return {"response":out[0]["generated_text"], "logits":self.logits_store.logits, "scores": self.logits_store.scores}
    
    def info(self):
        return {
            "name": self.name,
            "config": self.config,
            "prompting_hint":self.prompting_hint
        }
    


class LLM_Server:

    def __init__(self, wrapped_model:LLM_Server_Wrapper):
        self.wrapped_model = wrapped_model
        self.app = flask.Flask(__name__)

        @self.app.route("/request", methods=["POST"])
        def request():
            data = flask.request.json
            prompt = data["prompt"]
            process_logits = data.get("process_logits", False)
            if prompt is None:
                return "Prompt is required", 400
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            
            return self.wrapped_model.request(prompt, process_logits)
        
        @self.app.route("/info", methods=["GET"])
        def info():
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            return self.wrapped_model.info()
        
        @self.app.route("/", methods=["GET"])
        def ping():
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            return f"ping: {self.wrapped_model.name}"
    
    def start(self):
        self.app.run(port=5000)


if __name__ == "__main__":
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

    server = LLM_Server(LLM_Server_Wrapper("flan-t5-xxl", tokenizer, model, {}))
    server.start()


        



        



        