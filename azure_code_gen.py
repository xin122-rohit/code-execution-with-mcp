import os
import re
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def clean_code_block(code: str) -> str:
    """Remove Markdown code fences and extract Python code."""
    # Match ```python
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return raw code if no fence
    return code.strip()

def generate_code(query: str, context: str = "") -> str:
    prompt = f"""
You are an AI agent using MCP code execution. Solve using search_tools() and call_mcp_tool().

Previous context: {context}

User task: {query}

Write ONLY Python code. No explanations. No Markdown.

Use:
- search_tools(query, 'full') → returns list of tools
- call_mcp_tool(tool_name, args) → returns result
- Print final answer with print()

Example:
tools = search_tools('multiply')
result = call_mcp_tool(tools[0]['name'], {{'a': 5, 'b': 3}})
print(result)
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    raw_code = response.choices[0].message.content.strip()
    print(f"Raw LLM Output:\n{raw_code}\n")
    
    # Clean and return
    clean = clean_code_block(raw_code)
    print(f"Cleaned Code:\n{clean}\n")
    return clean