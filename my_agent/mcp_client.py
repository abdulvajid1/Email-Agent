import asyncio
from typing import cast
from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_SERVERS = {
        "web_search": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "fastmcp", "run", r"D:\AI\simpleai\my_agent\mcp_server.py"],
        }
    }


async def main():
    client = MultiServerMCPClient(MCP_SERVERS) # type: ignore
    tools = await client.get_tools()
    print("Available tools:", tools)


if __name__ == "__main__":
    asyncio.run(main())
