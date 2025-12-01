import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional, Tuple
 
import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import AzureOpenAI
 
load_dotenv()
 
langfuse = Langfuse()  # Auto-reads LANGFUSE_* env vars
 
AZURE_CLIENT = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
)
JUDGE_MODEL = os.getenv("AZURE_OPENAI_MODEL")
 
 
def calculate_cost(input_tokens: Optional[int], output_tokens: Optional[int]) -> Tuple[float, float]:
    INPUT_COST = 0.15 / 1_000_000
    OUTPUT_COST = 0.60 / 1_000_000
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    return input_tokens * INPUT_COST, output_tokens * OUTPUT_COST
 
 
def download_all_traces_to_csv(
    output_file: str = "my_langfuse_traces_nov2025.csv",
    days: int = 180,
    limit_per_page: int = 100,
    evaluate_scores: bool = False,
):
    all_traces = []
    page = 1
    from_dt = datetime.now(UTC) - timedelta(days=days)
 
    print(f"Downloading all traces from the last {days} days (since {from_dt.isoformat()}Z)...")
 
    while True:
        print(f"Fetching page {page} (limit={limit_per_page})...")
 
        response = langfuse.api.trace.list(
            page=page,
            limit=limit_per_page,
            from_timestamp=from_dt,
            order_by="timestamp.DESC",
            fields="core,io,metrics,scores",
        )
 
        if not response.data or len(response.data) == 0:
            print("No more traces. Done!")
            break
 
        print(f"  → Got {len(response.data)} traces (total so far: {len(all_traces) + len(response.data)})")
 
        for trace in response.data:
            input_tokens = getattr(trace, "input_tokens", None)
            output_tokens = getattr(trace, "output_tokens", None)
            input_cost = getattr(trace, "input_cost", None)
            output_cost = getattr(trace, "output_cost", None)
            total_cost = getattr(trace, "total_cost", None)
 
            if input_cost is None or output_cost is None or total_cost is None:
                fallback_input_cost, fallback_output_cost = calculate_cost(input_tokens, output_tokens)
                input_cost = input_cost if input_cost is not None else fallback_input_cost
                output_cost = output_cost if output_cost is not None else fallback_output_cost
                total_cost = total_cost if total_cost is not None else input_cost + output_cost
 
            row: Dict[str, Any] = {
                "trace_id": trace.id,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "name": trace.name or "",
                "user_id": trace.user_id or "",
                "session_id": trace.session_id or "",
                "input": trace.input,
                "output": trace.output,
                "model": getattr(trace, "model", None),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": getattr(trace, "total_tokens", None),
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "latency_seconds": getattr(trace, "latency", None),
                "metadata": str(trace.metadata) if trace.metadata else "",
                "tags": ", ".join(trace.tags) if hasattr(trace, "tags") and trace.tags else "",
                "release": trace.release or "",
                "version": trace.version or "",
            }
 
            if evaluate_scores:
                print(f"    Evaluating trace {trace.id}...")
                scores = evaluate_agent_trace(trace.id)
                for key, value in scores.items():
                    row[f"score_{key}"] = value
 
            all_traces.append(row)
 
        page += 1
 
    if all_traces:
        df = pd.DataFrame(all_traces)
        df.to_csv(output_file, index=False)
        total_cost = df["total_cost_usd"].sum()
        print(f"\nSUCCESS! Exported {len(df):,} traces → {output_file}")
        print(f"Total LLM cost in period: ${total_cost:,.6f}")
    else:
        print("No traces found.")
 
 
def fetch_full_trace(trace_id: str) -> Optional[Any]:
    """
    Retrieve a trace with full metadata/spans using the Langfuse Python SDK.
    Handles SDK variants where trace.get lives under langfuse.api.trace or elsewhere.
    """
    api_client = getattr(langfuse, "api", None)
    if api_client is not None:
        trace_resource = getattr(api_client, "trace", None)
        if trace_resource is not None and hasattr(trace_resource, "get"):
            try:
                return trace_resource.get(trace_id=trace_id)
            except Exception as exc:
                print(f"Failed to fetch trace via langfuse.api.trace.get: {exc}")
 
    if hasattr(langfuse, "trace") and callable(getattr(langfuse, "trace")):
        try:
            return langfuse.trace(trace_id)
        except Exception as exc:
            print(f"Failed to fetch trace via langfuse.trace: {exc}")
 
    print("Langfuse client does not expose a supported trace getter.")
    return None
 
 
def evaluate_agent_trace(trace_id: str) -> Dict[str, Any]:
    trace = fetch_full_trace(trace_id)
    if trace is None:
        print(f"Trace {trace_id} could not be fetched; skipping score calculation.")
        return {}
    question = extract_user_question(trace.input)
    final_answer = extract_final_answer(trace.output)
    full_conversation = extract_full_conversation(trace)
 
    scores: Dict[str, Any] = {}
 
    scores["goal_completion"] = llm_judge(
        prompt_template="""
Question: {question}
Final Answer: {answer}
 
Did the agent fully solve the user's original goal or question?
Score 0.0–1.0 (1.0 = completely solved, no missing parts)
Respond ONLY with valid JSON: {{"score": 0.XX, "explanation": "one short sentence"}}
        """,
        question=question,
        answer=final_answer,
    )
 
    scores["tool_selection_accuracy"] = llm_judge(
        prompt_template="""
Question: {question}
All tool calls made: {tools_used}
 
Did the agent select the correct tools and in a logical order?
Score 0.0–1.0
JSON only: {{"score": 0.XX, "explanation": "..."}}
        """,
        question=question,
        tools_used=extract_tool_calls(trace),
    )
 
    scores["reasoning_quality"] = llm_judge(
        prompt_template="""
Full conversation (including thoughts and tool results):
{conversation}
 
Are the intermediate reasoning steps logical, coherent, and free of major jumps?
Score 0.0–1.0
JSON only.
        """,
        conversation=full_conversation,
    )
 
    scores["hallucinated_tool_output"] = llm_judge(
        prompt_template="""
Question: {question}
Conversation with tool results: {conversation}
 
Did the agent ever invent or hallucinate a tool result that wasn't actually returned?
0.0 = yes (bad), 1.0 = no hallucinations
JSON only.
        """,
        question=question,
        conversation=full_conversation,
    )
 
    scores["conciseness"] = llm_judge(
        prompt_template="""
Final Answer: {answer}
 
Is the final response concise and to-the-point (no unnecessary fluff/repetition)?
Score 0.0–1.0
JSON only.
        """,
        answer=final_answer,
    )
 
    start = getattr(trace, "start_time", None) or getattr(trace, "timestamp", None)
    end = getattr(trace, "end_time", None) or datetime.now(UTC)
    latency = (end - start).total_seconds() if (start and end) else None
 
    scores["total_steps"] = count_agent_steps(trace)
    scores["latency_seconds"] = round(latency, 2) if latency is not None else None
    usage = getattr(trace, "usage", None)
    scores["total_tokens"] = (
        usage.total_tokens
        if usage and getattr(usage, "total_tokens", None) is not None
        else getattr(trace, "total_tokens", 0)
    )
    scores["task_success_binary"] = 1.0 if scores.get("goal_completion", 0) >= 0.90 else 0.0
 
    score_descriptions = {
        "goal_completion": "Did the agent fully solve the user's goal?",
        "tool_selection_accuracy": "Correct tools chosen and ordered?",
        "reasoning_quality": "Logical intermediate reasoning?",
        "hallucinated_tool_output": "1.0 = no fake tool results (lower = bad)",
        "conciseness": "Final answer not verbose",
        "total_steps": "Number of reasoning/tool steps (lower = more efficient)",
        "latency_seconds": "End-to-end latency in seconds",
        "total_tokens": "Total tokens used",
        "task_success_binary": "Strict pass/fail (goal_completion ≥ 0.9)",
    }
 
    for name, value in scores.items():
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value) if isinstance(value, (int, float)) else 0.0,
            comment=score_descriptions.get(name, ""),
            data_type="NUMERIC",
        )
 
    metric_summary = ", ".join(
        f"{name}={value:.3f}"
        for name, value in scores.items()
        if isinstance(value, (int, float))
    )
    print(f"Evaluated trace {trace_id} → {metric_summary}")
    return scores
 
 
def llm_judge(prompt_template: str, **kwargs) -> float:
    if not AZURE_CLIENT or not JUDGE_MODEL:
        print("Azure OpenAI configuration missing. Returning 0.0 score.")
        return 0.0
 
    full_prompt = prompt_template.strip().format(**kwargs)
 
    try:
        response = AZURE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=500,
            messages=[{"role": "system", "content": full_prompt}],
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return float(result.get("score", 0.0))
    except Exception as e:
        raw_content = locals().get("content", "N/A")
        print(f"Judge failed: {e}\nRaw: {raw_content}")
        return 0.0
 
 
def extract_user_question(input_data: Any) -> str:
    if isinstance(input_data, dict):
        message_fallback = ""
        if input_data.get("messages"):
            last_message = input_data["messages"][-1]
            message_fallback = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
        return (
            input_data.get("question")
            or input_data.get("input")
            or message_fallback
            or ""
        )
    return str(input_data or "")
 
 
def extract_final_answer(output_data: Any) -> str:
    if isinstance(output_data, dict):
        if "choices" in output_data:
            first_choice = output_data["choices"][0]
            return first_choice.get("message", {}).get("content", "")
        if "content" in output_data:
            return output_data["content"]
    return str(output_data or "")
 
 
def extract_tool_calls(trace) -> str:
    tools = []
    for span in getattr(trace, "spans", []) or []:
        name = getattr(span, "name", "")
        if name in ["tool", "function", "retriever"] or ("tool" in name.lower()):
            tools.append(f"- {name}: {getattr(span, 'input', '')}")
    return "\n".join(tools) or "No tools used"
 
 
def count_agent_steps(trace) -> int:
    spans = getattr(trace, "spans", []) or []
    llm_calls = len([s for s in spans if getattr(s, "type", "") == "llm"])
    tool_calls = len([s for s in spans if "tool" in getattr(s, "name", "").lower()])
    return llm_calls + tool_calls
 
 
def extract_full_conversation(trace) -> str:
    parts = []
    for span in getattr(trace, "spans", []) or []:
        if getattr(span, "input", None):
            parts.append(f"Thought/Step: {span.input}")
        if getattr(span, "output", None):
            parts.append(f"Result: {span.output}")
    return "\n\n".join(parts)[:15000]
 
 
if __name__ == "__main__":
    download_all_traces_to_csv(days=180, limit_per_page=100, evaluate_scores=True)