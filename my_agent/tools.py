from mcp_client import MCP_SERVERS
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(MCP_SERVERS)  # type: ignore
async def get_tools() -> list:
    web_tool = await mcp_client.get_tools(server_name="web_search")
    return web_tool
