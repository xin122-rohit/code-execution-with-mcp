import os
import pandas as pd
from datetime import datetime, timedelta
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse()  # Auto-reads LANGFUSE_* env vars

def download_all_traces_to_csv(
    output_file: str = "my_langfuse_traces_nov2025.csv",
    days: int = 180,
    limit_per_page: int = 100,
):
    all_traces = []
    page = 1
    from_dt = datetime.utcnow() - timedelta(days=days)

    print(f"Downloading all traces from the last {days} days (since {from_dt.isoformat()}Z)...")

    while True:
        print(f"Fetching page {page} (limit={limit_per_page})...")

        response = langfuse.api.trace.list(
            page=page,
            limit=limit_per_page,
            from_timestamp=from_dt,
            order_by="timestamp.DESC",
            # THIS IS THE KEY: explicitly ask for token usage & cost
            fields="core,io,metrics,scores",
            # core = id, timestamp, name, etc.
            # io    = input/output
            # metrics = "latency, totalCost, inputTokens, outputTokens, etc",
            # scores = optional
        )

        if not response.data or len(response.data) == 0:
            print("No more traces. Done!")
            break

        print(f"  → Got {len(response.data)} traces (total so far: {len(all_traces) + len(response.data)})")

        for trace in response.data:
            # Now these fields exist because we requested "metrics"
            all_traces.append({
                "trace_id": trace.id,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "name": trace.name or "",
                "user_id": trace.user_id or "",
                "session_id": trace.session_id or "",
                "input": trace.input,
                "output": trace.output,
                "model": getattr(trace, "model", None),  
                "input_tokens": getattr(trace, "input_tokens", None),
                "output_tokens": getattr(trace, "output_tokens", None),
                "total_tokens": getattr(trace, "total_tokens", None),
                "input_cost_usd": getattr(trace, "input_cost", None),
                "output_cost_usd": getattr(trace, "output_cost", None),
                "total_cost_usd": getattr(trace, "total_cost", None),
                "latency_seconds": getattr(trace, "latency", None),
                "metadata": str(trace.metadata) if trace.metadata else "",
                "tags": ", ".join(trace.tags) if hasattr(trace, "tags") and trace.tags else "",
                "release": trace.release or "",
                "version": trace.version or "",
            })

        page += 1

    # Save
    if all_traces:
        df = pd.DataFrame(all_traces)
        df.to_csv(output_file, index=False)
        total_cost = df["total_cost_usd"].sum()
        print(f"\nSUCCESS! Exported {len(df):,} traces → {output_file}")
        print(f"Total LLM cost in period: ${total_cost:,.6f}")
    else:
        print("No traces found.")

# RUN
if __name__ == "__main__":
    download_all_traces_to_csv(days=180, limit_per_page=100)
 