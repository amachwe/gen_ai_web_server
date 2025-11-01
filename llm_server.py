import flask
import abc
import transformers

class LogitStoreProcessor(transformers.LogitsProcessor):
    """
    A processor for storing logits and scores from language model predictions.

    This class is designed to be used as part of a logits processor list in huggingface transformer based language models.

    Attributes:
        logits (list): A list to store logits.
        scores (list): A list to store scores.

    Methods:
        There are no public methods defined in this excerpt.
    """
    def __init__(self, ):
        self.logits = []
        self.scores = []

    def clear(self):
        """
        Clear the stored logits and scores.
        """
        self.logits = []
        self.scores = []    

    def __call__(self, input_ids, scores):
        # Store logits and scores - nothing else...
        self.logits.append(input_ids.tolist())
        self.scores.append(scores.tolist())
        return scores

class Wrapper(abc.ABC):

    def request(self, prompt:list[dict], *args, **kwargs,)->dict:
        pass

    def info(self)->dict:
        pass

    def get_vocab(self)->dict:
        pass


class LLM_Server_Wrapper(Wrapper):
    """
    A wrapper class for a language model.

    Methods:
        request: Request a response from the language model wrapped.
        info: Get information about the language model wrapped.
    
    """
    def __init__(self, name:str, tokenizer, model, config:dict):
        """
        Parameters:
            name (str): The name of the language model.
            tokenizer: The tokenizer for the language model (from Transformers library).
            model: The language model to wrap (from Transformers library).
            config (dict): Configuration for the language model (including device, etc.)
        """
        self.name = name
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logits_store = LogitStoreProcessor()
        self.logits_processor_list = transformers.LogitsProcessorList([self.logits_store,])
        self.prompting_hint = config.get("prompting_hint", "")

        if config.get("device", "cpu") == "cuda":
            self.model.to("cuda")

    def print_kwargs(self, kwargs):
        """
        Print the kwargs dictionary.
        """
        for k,v in kwargs.items():
            print(f"{k}: {v}")

    def get_vocab(self):
        """
        Get the vocabulary of the language model.
        Returns:
            dict: A dictionary containing the vocabulary of the language model.
        """
        return self.model.get_vocab()

    def request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={})->dict:
        """
        Request a response from the language model.
        prompt (list[dict]): The prompt to send to the language model.
        process_logits (bool): Whether to process logits and scores from the language model (can add overhead to the request). Default=False
        run_config (dict): Run configuration for the request (including do_sample, etc.)

        Returns:
            dict: A dictionary containing the response, logits, and scores from the language model. Logits and scores are only returned if process_logits is True.
        """
        max_length = self.config.get("max_length", None)
        num_return_sequences = self.config.get("num_return_sequences", 1)
        output_scores = self.config.get("output_scores", None)
        max_new_tokens = self.config.get("max_new_tokens", 500)
        do_sample = run_config.get("do_sample", self.config.get("do_sample", False))
        temperature = run_config.get("temperature", self.config.get("temperature", 0.0))
        debug_mode = run_config.get("debug_mode", self.config.get("debug_mode", False))
  

        ## Encode the prompt using the tokenizer
        enc = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        
        ## Handling configuration options. Expand in future to improve tunability of LLMs.
        kwargs = {}

        kwargs["logits_processor"] = []
        
        if self.logits_processor_list:
            for p in self.logits_processor_list:
                print("Clearing processor")
                p.clear()

        if process_logits:   
            kwargs["logits_processor"] = self.logits_processor_list
        
        if max_length:
            kwargs["max_length"] = max_length

        if output_scores:
            kwargs["output_scores"] = output_scores

       
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["num_return_sequences"] = num_return_sequences
        kwargs["do_sample"] = do_sample
        kwargs["temperature"] = temperature


        if debug_mode:
            self.print_kwargs(kwargs)
            print("Process Logits: ", process_logits)
            print(prompt)

        ## Call generate method of the wrapped model
        res = self.model.generate(enc, **kwargs)
       
        dec = self.tokenizer.decode(res[0], skip_special_tokens=True)
        
        return {"response":dec, "logits":self.logits_store.logits, "scores": self.logits_store.scores}
    
    def info(self)->dict:
        """
        Get information about the model being wrapped.
        Returns:
            dict: A dictionary containing the name, configuration, and prompting hint for the wrapped model.
        """
        return {
            "name": self.name,
            "config": self.config,
            "prompting_hint":self.prompting_hint
        }
    

class LLM_Server_Pipe_Wrapper(Wrapper):
    """
    A wrapper class for a language model. Uses pipelines for text generation.

    Methods:
        request: Request a response from the language model wrapped.
        info: Get information about the language model wrapped.
    
    """

    def __init__(self, name, tokenizer, model, config:dict):
        """
        Parameters:
            name (str): The name of the language model.
            tokenizer: The tokenizer for the language model (from Transformers library).
            model: The language model to wrap (from Transformers library).
            config (dict): Configuration for the language model (including device, etc.)
        """
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
    
    def print_kwargs(self, kwargs):
        """
        Print the kwargs dictionary.
        """
        for k,v in kwargs.items():
            print(f"{k}: {v}")

    def get_vocab(self):
        """
        Get the vocabulary of the language model.
        Returns:
            dict: A dictionary containing the vocabulary of the language model.
        """
        return self.model.get_vocab()
    

    def request(self, prompt:list[dict], process_logits:bool=False, run_config:dict={})->dict:
        """
        Request a response from the language model.
        prompt (list[dict]): The prompt to send to the language model.
        process_logits (bool): Whether to process logits and scores from the language model (can add overhead to the request). Default=False
        run_config (dict): Run configuration for the request (including do_sample, etc.)
        
        Returns:
            dict: A dictionary containing the response, logits, and scores from the language model. Logits and scores are only returned if process_logits is True.
        """
        max_length = self.config.get("max_length", None)
        num_return_sequences = self.config.get("num_return_sequences", 1)
        output_scores = self.config.get("output_scores", None)
        max_new_tokens = self.config.get("max_new_tokens", 500)
        do_sample = run_config.get("do_sample", self.config.get("do_sample", False))
        temperature = run_config.get("temperature", self.config.get("temperature", 0.0))
        debug_mode = run_config.get("debug_mode", self.config.get("debug_mode", False))
        
        
        ## Handling configuration options. Expand in future to improve tunability of LLMs.
        kwargs = {}

        kwargs["logits_processor"] = []
        
        if self.logits_processor_list:
            for p in self.logits_processor_list:
                print("Clearing processor")
                p.clear()

        if process_logits:      
            kwargs["logits_processor"] = self.logits_processor_list
        
        if max_length:
            kwargs["max_length"] = max_length

        if output_scores:
            kwargs["output_scores"] = output_scores

       
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["num_return_sequences"] = num_return_sequences
        kwargs["do_sample"] = do_sample
        kwargs["temperature"] = temperature
  

        if debug_mode:
            self.print_kwargs(kwargs)
            print("Process Logits: ", process_logits)
            print(prompt)

        ## Using transformers pipelines for text generation instead of directly calling generate method.
        pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.model.device)
        out = pipe(prompt, **kwargs)
        
        return {"response":out[0]["generated_text"], "logits":self.logits_store.logits, "scores": self.logits_store.scores}
    
    def info(self):
        """
        Get information about the model being wrapped.
        Returns:
            dict: A dictionary containing the name, configuration, and prompting hint for the wrapped model.
        """
        return {
            "name": self.name,
            "config": self.config,
            "prompting_hint":self.prompting_hint
        }
    


class LLM_Server:
    """
    Main LLM Server class for serving a language model.
    """
    def __init__(self, wrapped_model:Wrapper, port:int=5000):
        """
        Initiate the server with a wrapped language model.
        Parameters:
            wrapped_model (LLM_Server_Wrapper): The wrapped language model to serve.
            port (int): The port to serve the language model on. Default=5000
        """
        self.wrapped_model = wrapped_model
        self.port = port
        self.app = flask.Flask(__name__)

        @self.app.route("/vocab", methods=["GET"])
        def vocab():
            """
            Request handler for the vocab method of the wrapped language model.
            """
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            
            return self.wrapped_model.get_vocab()

        @self.app.route("/request", methods=["POST"])
        def request():
            """
            Request handler for the server. Handles requests for responses from the language model.
            """
            data = flask.request.json
            prompt = data["prompt"]
            process_logits = data.get("process_logits", False) ## Optional parameter to process logits and scores from the language model.
            run_config = data.get("run_config", {})
            if prompt is None:
                return "Prompt is required", 400
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            
            return self.wrapped_model.request(prompt, process_logits, run_config)
        
        @self.app.route("/info", methods=["GET"])
        def info():
            """
            Request handler for the info method of the wrapped language model.
            """
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            return self.wrapped_model.info()
        
        @self.app.route("/", methods=["GET"])
        def ping():
            """
            Method to check if the server is running.
            """
            if self.wrapped_model is None:
                return "Model is not loaded", 500
            return f"ping: {self.wrapped_model.name}"
    
    def start(self):
        """
        Call this method to start the server on the configured port (default=5000)
        """
        self.app.run(port=self.port)


if __name__ == "__main__":
    ## Example usage with Google Flan T5 being wrapped. ##
    import transformers

    print("Starting EXAMPLE LLM SERVER... are you sure you meant to run this script?")

    ## 1. Load your favourite model and tokenizer using Transformers library.
    MODEL_ID = "google/flan-t5-base"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    ## 2. Create a server with the wrapped model.
    server = LLM_Server(LLM_Server_Wrapper(MODEL_ID, tokenizer, model, {}))

    ## 3. Start the server (default port = 5000)
    server.start()

        



        