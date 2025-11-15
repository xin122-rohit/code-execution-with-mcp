# mcp_code_agent.py
from azure_code_gen import generate_code
from repl import MCPREPL
import os

class MCPCodeAgent:
    def __init__(self):
        self.repl = MCPREPL()
        self.history = []  # For state (article-style persistence)

    def run(self, query: str) -> str:
        print(f"\nYou: {query}")

        # Step 1: Initial discovery (progressive: names first)
        initial_tools = self.repl.search_tools(query, "name")
        print(f"Discovered {len(initial_tools)} tool names")

        # Step 2: Generate code with Azure
        context = "\n".join([f"History: {h}" for h in self.history[-2:]])  # Last 2 for context
        code = generate_code(query, context)
        print(f"Generated Code:\n{code}")

        # Step 3: Execute in REPL
        output = self.repl.execute_code(code)
        print(f"Execution Output: {output}")

        # Step 4: Extract final answer (from prints/logs)
        final_answer = self.extract_answer(code, output)  # Custom logic or parse prints
        self.history.append(f"Query: {query} → Answer: {final_answer}")
        return final_answer

    def extract_answer(self, code: str, output: str) -> str:
        # Simple: Look for last print (article-style filtering)
        lines = output.split("\n")
        return lines[-1] if lines else "No answer generated"

# Run Interactive
if __name__ == "__main__":
    agent = MCPCodeAgent()
    print("MCP Code Execution Agent Ready! (Article-Style with Azure OpenAI)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        answer = agent.run(q)
        print(f"Agent: {answer}\n")