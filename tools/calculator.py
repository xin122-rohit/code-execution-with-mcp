import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncpg
import os
from typing import List, Dict, Any
from typing import Annotated
load_dotenv()

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))        # insert at beginning = safer

# This import will now work every single time
from initialize_server import mcp
def register_tool(mcp):
    def decorator(func):
        return mcp.tool()(func)
    return decorator

@register_tool(mcp)
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

@register_tool(mcp)
def subtract_numbers(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@register_tool(mcp)
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@register_tool(mcp)
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b (returns 0.0 if dividing by zero)."""
    if b == 0:
        return 0.0
    return a / b
