from typing import Any, Dict, Iterator, List, Optional
from langchain_core.runnables import Runnable
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.output_parsers import PydanticOutputParser
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
from pydantic import BaseModel, Field
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
    bound_tools: Optional[Any] = None       
    structured_schema: Optional[Any] = None
    n_retry: int = 5 
    
    def __init__(self, model: str = 'llama3.1', bound_tools=None, structured_schema=None, n_retry: int = 5):
        super().__init__()
        self.bound_tools = bound_tools
        self.structured_schema = structured_schema
        api_key: str = os.environ['API_KEY']
        base_url: str = os.environ['MODEL_URL']
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.n_retry = n_retry

    def _generate(self, messages, stop=None, callbacks=None, **kwargs):
        if self.bound_tools:
            return self._call_model_with_tool(messages)
        if self.structured_schema:
            return self._call_model_with_schema(messages)
        
        messages = convert_to_openai_messages(messages)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            max_tokens=self.max_tokens,
        )
        message = AIMessage(content=response.choices[0].message.content, additional_kwargs={})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages: BaseMessage, stop=None, run_manager=None, **kwargs):
        messages = convert_to_openai_messages(messages)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            max_tokens=self.max_tokens,
        )
        for chat in response:
            delta_content = chat.choices[0].delta.content
            if delta_content is None:   # ✅ last chunk has None content, skip it
                continue
            message_chunk = AIMessageChunk(content=delta_content, additional_kwargs={})
            chunk = ChatGenerationChunk(message=message_chunk)
            yield chunk


    @property
    def _llm_type(self) -> str:
        return self.model

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "model_name": self.model,
        }

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.__class__(
            model=self.model,         
            bound_tools=formatted_tools
        )

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        return self.__class__(
            model=self.model,           
            bound_tools=None,
            structured_schema=schema
        )

    def _format_messages_for_tools(self, messages: list):
        messages = messages.copy()
        tool_system_instruction = (
            f"You are a helpful AI assistant with tool calling ability.\n"
            f"Use a tool when the user's request matches one. Otherwise respond normally.\n"
            f"When calling a tool respond ONLY with a JSON object in this exact format:\n"
            f'{{"name": "<function_name>", "parameters": {{"arg": "value"}}}}\n'
            f"Do not include any other text when calling a tool.\n\n"
            f"Available tools:\n{json.dumps(self.bound_tools, indent=2)}"
        )

        # Change the current system prompt instruction to tool calling instruction
        system_prompt_changed = False
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg.content = tool_system_instruction
                system_prompt_changed = True

        # if system prompt not found, add system_message with new instruction        
        if not system_prompt_changed:
            messages.insert(0, SystemMessage(content=tool_system_instruction))
        return messages

    def _call_model_with_tool(self, messages):
        retry_left = self.n_retry
        
        # setup of tool instruction system prompt
        messages = self._format_messages_for_tools(messages)

        while retry_left > 0:
            retry_left -= 1 
            openai_messages = convert_to_openai_messages(messages)
            response = self.client.chat.completions.create(
                messages=openai_messages,
                model=self.model,
                stream=False,
                max_tokens=self.max_tokens,
            )
            raw_output = response.choices[0].message.content.strip()
            json_match = re.search(r'^\s*\{[\s\S]*\}\s*$', raw_output) # there can be a bug here, what if we don't match any json

            if json_match:
                try:
                    tool_json = json.loads(json_match.group(0))
                    message = AIMessage(
                                content='',
                                tool_calls=[
                                    ToolCall(
                                        name=tool_json.get("name"),
                                        args=tool_json.get("parameters", {}),
                                        id=str(uuid.uuid4())
                            )
                        ]
                    )   
                    return ChatResult(generations=[ChatGeneration(message=message)])
                          
                except json.JSONDecodeError as e:
                    human_content = f"""The output you given has error when i try to parse it:,
                    ### The output you given:
                    {raw_output}
                    ### Error
                    {e}
                    please try one more time and i will give your raw output for json parsing"""
                    
                    messages.extends([
                        AIMessage(content=raw_output),
                        HumanMessage(content=human_content)
                    ])
        
            else: # if json parsing couldn't find any json match, retry with issue
                human_content = f"""The output you given has issue while parsing to json, re.search(r'^\s*\{{[\s\S]*\}}\s*$', your_output) couldn't match any match:
                ### Your output:
                {raw_output}
                please try one more time."""
                messages.extend([
                    AIMessage(content=raw_output),
                    HumanMessage(content=human_content)
                ])
        
        # message = AIMessage(content=raw_output)
        message = AIMessage(content=f"Tool parsing failed.\n\nLast output:\n{raw_output}")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _call_model_with_schema(self, messages):
        pydantic_parser = None

        # Must check it's a class first before calling issubclass
        if isinstance(self.structured_schema, type) and issubclass(self.structured_schema, BaseModel):
            pydantic_parser = PydanticOutputParser(pydantic_object=self.structured_schema)
            schema = self.structured_schema.model_json_schema()
        else:
            schema = self.structured_schema

        schema_instruction = f"""You are a structured data extraction assistant.
Read the user's input and produce a JSON object that matches the given schema.

RULES:
- Output MUST be valid JSON — a single JSON object
- Do NOT include any text, markdown, or code blocks before/after the JSON
- Fill fields from user input; use null if a field cannot be determined
- Do NOT invent extra fields

SCHEMA:
{json.dumps(schema, indent=2)}"""

        # ✅ Only system prompt + last human message — no conversation history
        messages = [
            SystemMessage(content=schema_instruction),
            messages[-1]
        ]
        messages = convert_to_openai_messages(messages)

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            max_tokens=self.max_tokens,
        )
        raw_output = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps output anyway
        fenced = re.match(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
        if fenced:
            raw_output = fenced.group(1).strip()

        if pydantic_parser:
            try:
                pydantic_object = pydantic_parser.parse(text=raw_output)
                # ✅ AIMessage content must be str — serialize the pydantic object
                message = AIMessage(
                    content=pydantic_object.model_dump_json(),
                    additional_kwargs={"parsed": pydantic_object}  # caller can grab the object here
                )
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as error:  # ✅ was: except Exception error (syntax error)
                print(f"Pydantic parsing failed: {error}, falling back to raw JSON")

        # Fallback: return raw parsed dict
        try:
            parsed = json.loads(raw_output)
            # ✅ Return parsed JSON as AIMessage — NOT a ToolCall (was copy-pasted wrongly)
            message = AIMessage(
                content=json.dumps(parsed),
                additional_kwargs={"parsed": parsed}
            )
        except json.JSONDecodeError:
            message = AIMessage(
                content=raw_output,
                additional_kwargs={"schema_parse_error": True}
            )

        return ChatResult(generations=[ChatGeneration(message=message)])
        
    
    
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