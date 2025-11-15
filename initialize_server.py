from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
load_dotenv()

mcp = FastMCP("flight server")
mcp.settings.host = "127.0.0.1"
mcp.settings.port = 9000
app =FastMCP()