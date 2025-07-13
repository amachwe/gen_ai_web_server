from pydantic import  Field
from typing import Optional, List, Any, Dict, Sequence, Union, Callable, Literal
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage)
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.language_models import LanguageModelInput
import json 
import logging


logging.basicConfig(filemode="w", filename=f"log.txt", level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainCustomModel(BaseChatModel):
    
    """Custom Model wrapped by Gen AI Web Server"""

    model_name: str = Field(default="custom", alias="model", description="Name of the custom model")
    client: Any = Field(default=None, description="Client to send requests to the custom model")
    tools: Optional[Sequence[Dict[str, Any]]] = Field(
        default=[], 
        description="Tools to be used by the custom model"
    )
    tool_prompt: str = Field(
        default="",
        description="Prompt to be used for the tools"
    )
    

    def __init__(self, _client):
        super().__init__()
        self.client = _client
        self.tools = None
        self.tool_prompt = ""

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        """
        Generate a response from the custom model.
        """
        # Build the static tool description part of the prompt.
        if self.tools and self.tool_prompt == "":
            tool_prompt = """
            \nRespond to the user directly or use tools where appropriate. To use tool, return JSON with key name: name of the tool, key parameter: parameters mapped to variables and nothing else. 
            Available tools: """
            for tool in self.tools:
                tool_prompt += f"Name:{tool.name} \n Description:{tool.description} \nSchema:{tool.input_schema.model_json_schema()} "

        
        #Assemble the prompt for the LLM
        prompt = [{"role": "user", "content": messages[0].content + tool_prompt}]  

        #Opening request to the LLM (user -> LLM)
        response = self.client.send_request(prompt)

        # Extracting the text
        text = self.client.extract_response(response)
        while self.tools:
            try:
                text = text.replace("```json", "")
                text = text.replace("```", "")
                # Parsing the JSON response - otherwise it is not a tool call
                tool_data = json.loads(text)
                tool_name = tool_data.get("name")
                args = tool_data.get("parameters", {})
                logger.info(f"Tool Name: {tool_name}, Args: {args}")

                # Find the correct tool match 
                _tool = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        _tool = tool
                        break

                #Invoke the matched tool.
                tool_response = StructuredTool.from_function(_tool.func).invoke(args)
                logger.info(f"Tool Response: {tool_response}")

                # Result from the tool is put in a prompt
                tool_res = f"\nOriginal Message:\n {messages[0].content}\nTool Response: {tool_response}"# or do I use further tools?\n{tool_prompt} "
                logger.info(f"Tool Response: {tool_name} -- {tool_res}")

                # Send the tool result prompt back to the LLM
                response = self.client.send_request([{"role": "user", "content": tool_res}])
                text = self.client.extract_response(response)
                logger.info(f"Tool Response: {text}")
            except json.JSONDecodeError:
                break
            

        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])
    

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], type, Callable, BaseTool] 
        ],
        *,
        tool_choice: Optional[Union[str, Literal["any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        
        """
        Bind tools to the model.
        """

        self.tools = tools
        
        # for tool in tools:
        #     print(tool.)
        #     print(tool.name)
        #     print(tool.description)
        #     print(tool.input_schema.model_json_schema())
        return self
    
    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "custom"
                  
