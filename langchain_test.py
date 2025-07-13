import langchain_custom_model as  cm
from llm_client import Client
import langchain_core.tools as tools
import langchain_google_genai as genai
import os

KEY = os.environ["API_KEY"]

@tools.tool
def average(x:float, y:float) -> float:
    """
    Add Two numbers and return the average
    x: float, y: float
    Returns the average of x and y
    """

    return (x+y)/2

@tools.tool
def move(x:float, y:float, distance: float) -> tuple[float, float]:
    """
    Move a point x, y by a distance diagnonally
    x: float, y: float, distance: float
    Returns a tuple of new x and y coordinates
    """

    return (x+distance, y+distance)

@tools.tool
def reverse(x:str) -> str:
    """
    Reverse a string
    x: str
    Returns the reversed string
    """
    return x[::-1]

def get_text(response)->str:
    return response.content

if __name__ == "__main__":
    model = cm.LangChainCustomModel(Client())
    model = model.bind_tools([average, move, reverse])


    print(get_text(model.invoke("What is the average of 10 and 20?")))

    print(get_text(model.invoke("Move point (10, 20) by 5 units.")))

    print(get_text(model.invoke("Reverse the string: I love programming.")))


#print(model.invoke("Take the average of 10 and 5, then convert average to a point with same x,y then move by 5 units diagonally, and finally reverse the result."))

