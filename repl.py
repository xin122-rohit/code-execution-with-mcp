# repl.py
import json
from typing import Dict, Any, List
import httpx
from RestrictedPython import compile_restricted, safe_globals, utility_builtins
from RestrictedPython.Guards import safer_getattr

# === MCP CONFIG ===
MCP_BASE = "http://127.0.0.1:9000"
SEARCH_URL = f"{MCP_BASE}/search"
CALL_URL = f"{MCP_BASE}/call"


# # === SAFE GETITEM ===
def safe_getitem(obj, index):
    return obj[index]


from RestrictedPython import compile_restricted, safe_globals, utility_builtins
from RestrictedPython.Guards import safer_getattr
from RestrictedPython.PrintCollector import PrintCollector

class MCPREPL:
    def __init__(self):
        self.globals = safe_globals.copy()
        self.globals.update(utility_builtins)

        self.globals.update({
            "_print_": PrintCollector, 
            "_getitem_": safe_getitem,
            "_getattr_": safer_getattr,
            "httpx": httpx,
            "json": json,
            "str": str,
            "int": int,
            "float": float,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
        })


        self.namespace = {}

        # Your MCP tools
        self.globals["call_mcp_tool"] = self.call_mcp_tool
        self.globals["search_tools"] = self.search_tools

        self.namespace = {}

    def search_tools(self, query: str, detail: str = "full") -> List[Dict]:
        try:
            resp = httpx.get(SEARCH_URL, params={"q": query, "detail_level": detail})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        try:
            payload = {"tool": tool_name, "input": args}
            resp = httpx.post(CALL_URL, json=payload)
            resp.raise_for_status()
            result = resp.json().get("content")
            print(f"MCP Result: {result}")
            return result
        except Exception as e:
            print(f"MCP call error: {e}")
            return None

    def execute_code(self, code: str) -> str:
        try:
            byte_code = compile_restricted(code, '<string>', 'exec')
            exec(byte_code, self.globals, self.namespace)
            return "Code executed."
        except Exception as e:
            return f"Code error: {e}"