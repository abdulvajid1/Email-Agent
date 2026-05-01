from typing import Any, Dict, Iterator, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    HumanMessageChunk,
    ToolCall
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from langchain_core.messages import convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool
import json
import re
import uuid

class OpenAIGPT(BaseChatModel):
    
    model: str = Field(default='Meta/Llama3.1-8B-Instruct')
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 256 
    client: OpenAI = Field(default=None, exclude=True)
    bound_tools: dict = None
    
    def __init__(self, bound_tools=None):
        super().__init__()
        self.bound_tools = bound_tools
        api_key: str = "EMPTY"
        base_url: str = os.environ['MODEL_KEY']
    
        self.client = OpenAI(
        api_key=api_key,
        base_url=base_url)
        
        models = self.client.models.list()
        self.model = models.data[0].id
        
    def _generate(self, messages, stop = None, callbacks = None, **kwargs):
        
        if self.bound_tools:
            return self._call_model_with_tool(messages)
            
        messages = convert_to_openai_messages(messages)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            max_tokens=self.max_tokens,
        )
            
            
        message = AIMessage(content=response.choices[0].message.content, additional_kwargs = {}, my_meta_data = {})
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _stream(self, messages:BaseMessage, stop = None, run_manager = None, **kwargs):
            
        messages = convert_to_openai_messages(messages)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            max_tokens=self.max_tokens,
        )
        
        for chat in response:
            message_chunk = AIMessageChunk(content=chat.choices[0].delta.content, additional_kwargs = {}, my_meta_data = {})
            chunk = ChatGenerationChunk(message=message_chunk, usage_metadata={})
            yield chunk
            
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model
     
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.
        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "model_name": self.model,
        }
    
    def bind_tools(self, tools, *, tool_choice = None, **kwargs):
        formatted_tools = []
        for tool in tools:
            tool_json = convert_to_openai_tool(tool)
            formatted_tools.append(tool_json)
            
        return self.__class__(
            bound_tools = formatted_tools
        )
        
    def _format_messages_for_tools(self, messages): 
        tool_system_instruction = f"""You are a helpfull AI assistent, you also have tool calling ability whenever needed (user request is aligned with tool you provided) otherwise answer as helpfull assistent
You will Given the following functions and Use whenver function calling is necassary else respond normally as assistent, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. Do not use variables.\n\n{self.bound_tools}\n\n"""
        system_prompt_changed = False
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg.content = tool_system_instruction
                system_prompt_changed = True
        
        if not system_prompt_changed:
            messages.insert(0, SystemMessage(content=tool_system_instruction))
        
        return messages
    
    def _call_model_with_tool(self, messages):
        messages = self._format_messages_for_tools(messages)
        messages = convert_to_openai_messages(messages)

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            max_tokens=self.max_tokens,
        )

        raw_output = response.choices[0].message.content.strip()

        # --- Detect if the model produced a JSON tool call ---
        json_match = re.search(r'^\s*\{[\s\S]*\}\s*$', raw_output)
        if json_match:
            # Try to parse as JSON
            try:
                tool_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Handle single quotes, etc.
                try: 
                    tool_json = json.loads(json_match.group(0).replace("'", '"'))
                except json.JSONDecodeError:
                    return ChatResult(ChatGeneration(AIMessage(content=raw_output)))

            # Build ToolCall message
            message = AIMessage(
                content='',  # content empty because model responded via tool call
                tool_calls=[
                    ToolCall(
                        name=tool_json.get("name"),
                        args=tool_json.get("parameters", {}),
                        id=f'{uuid.uuid4()}'
                    )
                ]
            )
        else:
            # Normal text output (no tool call)
            message = AIMessage(content=raw_output)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
        
        
    
    
def test():
    gpt = OpenAIGPT()
    # tools
    
    @tool
    def add(a: int, b: int) -> int:
        "add two numbers"
        return a+b
    
    model_with_tool = gpt.bind_tools([add])
    output = model_with_tool.invoke("what is 20 and 35")
    print(output)
    
    
if __name__ == "__main__":
    test()