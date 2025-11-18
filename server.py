import os
import json
import asyncio
import importlib
import importlib.util
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from initialize_server import mcp
from starlette.requests import Request
from starlette.responses import JSONResponse
load_dotenv()

PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR"))
PLUGINS_DIR.mkdir(exist_ok=True)
# Store loaded modules for reloading
loaded_modules = {}
class PluginFileHandler:
    def __init__(self, server):
        self.server = server
    @staticmethod
    def load_all_plugins(server):
        print(f"Checking PLUGINS_DIR: {PLUGINS_DIR}")
        if PLUGINS_DIR.exists():
            py_files = list(PLUGINS_DIR.rglob("*.py"))
            print(f"Found files: {[f.relative_to(PLUGINS_DIR) for f in py_files]}")
            for file in py_files:
                if file.name in {"temp.py", "__init__.py"}:
                    continue
                print(f"Loading plugin: {file.name}")
                try:
                    spec = importlib.util.spec_from_file_location(file.stem, file)
                    if spec is None or spec.loader is None:
                        print(f"  → No spec/loader for {file.name}")
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)          # ← execute the module
                    loaded_modules[file.stem] = module
                    print(f"{loaded_modules}this is being loaded rohit")
                    

                    if hasattr(module, "register_tool"):
                        module.register_tool(server)
                        print(f"  → Registered tools from {file.name}print this tppp jviabfvhbandfv javerufvba")
                    else:
                        print(f"  → Skipped {file.name} (no register_tool function)")
                except Exception as e:
                    print(f"  → Failed to load {file.name}: {e}")
        else:
            print(f"PLUGINS_DIR does not exist: {PLUGINS_DIR}")


        
        
ALL_TOOLS: list[dict] = []

# for name, tool in mcp.list_tools():
#     print("-", name)


async def collect_tools_for_search():
    global ALL_TOOLS
    ALL_TOOLS = []

    tools = await mcp.list_tools()
    for t in tools:
        ALL_TOOLS.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
                "title":t.title,
            })
        
    print(ALL_TOOLS)
# === IMPORTS (MUST BE AT TOP) ===
@mcp.custom_route("/search", methods=["GET"])
async def search_tools(request: Request) -> JSONResponse:
    """Search tools by keyword."""
    params = request.query_params
    q = params.get("q", "").lower()
    detail_level = params.get("detail_level", "name")

    results = [
        tool for tool in ALL_TOOLS
        if q in tool["name"].lower() or 
           (tool.get("description") and q in tool["description"].lower())
    ][:50]

    if detail_level == "name":
        data = [{"name": t["name"]} for t in results]
    elif detail_level == "desc":
        data = [{"name": t["name"], "description": t.get("description", "")} for t in results]
    else:
        data = results

    return JSONResponse(data)
# === CUSTOM /call ROUTE (BYPASSES MCP PATH) ===
@mcp.custom_route("/call", methods=["POST"])
async def custom_call(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        tool_name = body["tool"]
        tool_input = body.get("input", {})

        # Call the tool
        result = await mcp.call_tool(tool_name, tool_input)
        
        # EXTRACT PLAIN TEXT
        if isinstance(result, tuple):
            content = result[0]
        elif hasattr(result, "text"):
            content = result[0].text
        else:
            content = result

        return JSONResponse({"content":content[0].text})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

if __name__ == "__main__":
    PluginFileHandler.load_all_plugins(mcp)
    asyncio.run(collect_tools_for_search())
    for route in getattr(mcp, "_custom_starlette_routes", []):
        print(f"  → {route.path} [{', '.join(route.methods)}]")
    
    print(f"\nMCP Server starting with {len(ALL_TOOLS)} tools.")
    print("Registered custom routes:")
    
    mcp.run(transport="streamable-http")