from sre_parse import State

from langchain_ollama import ChatOllama
from langchain_core.messages import (SystemMessage, 
                                     HumanMessage, 
                                     AIMessage, AnyMessage)

from langgraph.graph import START, StateGraph, add_messages, END
from typing import TypedDict

llm = ChatOllama(model="qwen3.5", reasoning=True)

# State definition
class MyState(TypedDict):
    messages: list[AnyMessage, add_messages]

# NODES   
def chat_node(state: MyState):
    messages = state.get("messages", [])
    if not messages:
        assert False, "No messages found in state."
    response = llm.invoke(messages)
    return {'messages': response}



graph = StateGraph(MyState)
graph.add_node("ChatNode", chat_node)
graph.add_edge(START, "ChatNode")
graph.add_edge("ChatNode", END)

graph = graph.compile()

if __name__ == "__main__":
    # Example usage
    state = MyState()
    state['messages'] = [SystemMessage(content="You are a helpful assistant."), 
                         HumanMessage(content="Hello, how are you?")]
    
    result = graph.stream(state, stream_mode='messages')

    thinking = False
    for i in result:
        if i[0].additional_kwargs.get("reasoning_content"):
            if not thinking:
                print("Thinking...")
                thinking = True
            print(i[0].additional_kwargs.get("reasoning_content"), end="", flush=True)
    
        if i[0].content:
            if thinking:
                print("\nDone thinking.")
                thinking = False
        
        print(i[0].content, end="", flush=True)
    # print(result)




    