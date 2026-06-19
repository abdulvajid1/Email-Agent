from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("Demo Agent")
web_search = DuckDuckGoSearchRun()


@mcp.tool
def search_web(query: str) -> str:
    """Search the web for the given query and return the results."""
    return web_search.invoke(query)


if __name__ == "__main__":
    mcp.run()  # default transport is stdio
