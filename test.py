import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from openai import AzureOpenAI
from langfuse import observe
# from langfuse.api import trace
# trace.Traces()
# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
)
PRICING_PER_MILLION = {
    "gpt-4o-mini":     {"input": 0.150, "output": 0.600},
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-35-turbo":    {"input": 0.50,  "output": 1.50},
}

CSV_FILE = "azure_openai_calls_india.csv"

def append_to_csv(record: dict):
    """Append a single record to CSV (creates file + headers if needed)"""
    df_new = pd.DataFrame([record])
    
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(CSV_FILE, index=False)
# === Price per 1M tokens (update these for your exact deployment) ===
# Example for gpt-4o-mini (as of Nov 2025, India region prices are same as global)
PRICING = {
    "gpt-4o-mini":     {"input": 0.150, "output": 0.600},
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-35-turbo":    {"input": 0.50,  "output": 1.50},
}

# Local file to store usage
USAGE_FILE = "azure_openai_usage.json"

def load_usage():
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    return []

def save_usage(records):
    with open(USAGE_FILE, "w") as f:
        json.dump(records, f, indent=2)

@observe()
def hello_llm():
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant that gives me detailed summary of a topic in 1000 words paragraph."},
            {"role": "user",   "content": "tell me about india"}
        ],
        temperature=0,
    )

    # Extract usage
    usage = response.usage
    model = response.model

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Calculate cost
    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

    # Store locally
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6),
    }
    append_to_csv(record)


    records = load_usage()
    records.append(record)
    save_usage(records)

    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens → Prompt: {prompt_tokens} | Completion: {completion_tokens} | Cost: ${cost:.6f}")

    return response.choices[0].message.content

# Run
if __name__ == "__main__":
    hello_llm()

    # Flush Langfuse
    from langfuse import get_client
    get_client().flush()

    # Bonus: print total cost so far
    total_cost = sum(r["cost_usd"] for r in load_usage())
    print(f"\nTotal cost recorded locally: ${total_cost:.6f}")