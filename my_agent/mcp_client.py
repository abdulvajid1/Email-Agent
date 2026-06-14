from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio


MCP_SERVERS = {
    "web_search": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run",
                 "fastmcp",
                 "run",
                 "D:\\AI\\simpleai\\my_agent\\mcp_server.py"],
    }
}

async def main():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print("Available tools:", tools)



if __name__ == "__main__":
    asyncio.run(main())